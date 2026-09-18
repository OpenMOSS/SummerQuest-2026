from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - Triton is only exercised on CUDA hosts.
    triton = None
    tl = None


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Unfused QK^T -> scale -> causal mask -> softmax -> PV baseline."""
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if is_causal:
        nq, nk = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(nq, device=q.device)[:, None]
        k_pos = torch.arange(nk, device=q.device)[None, :]
        scores = scores.masked_fill(q_pos < k_pos, float("-inf"))
    return torch.softmax(scores, dim=-1) @ v


def flash_attention_torch(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
    """Reference tiled FlashAttention computation using ordinary PyTorch ops."""
    if q.ndim < 2 or k.ndim != q.ndim or v.ndim != k.ndim:
        raise ValueError("q, k, and v must have matching leading dimensions")
    if q.shape[:-2] != k.shape[:-2] or k.shape[:-2] != v.shape[:-2]:
        raise ValueError("q, k, and v must share leading dimensions")
    if q.shape[-1] != k.shape[-1] or k.shape[-2] != v.shape[-2]:
        raise ValueError("incompatible attention dimensions")

    nq, nk, d = q.shape[-2], k.shape[-2], q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    q_flat = q.reshape(-1, nq, d)
    k_flat = k.reshape(-1, nk, d)
    v_flat = v.reshape(-1, nk, v.shape[-1])
    outputs = []
    lses = []

    # Tile over the key dimension while maintaining the online softmax state.
    q_tile = min(64, nq)
    k_tile = min(64, nk)
    for q_start in range(0, nq, q_tile):
        q_chunk = q_flat[:, q_start : q_start + q_tile]
        rows = q_chunk.shape[-2]
        m = torch.full((q_flat.shape[0], rows), -torch.inf, dtype=torch.float32, device=q.device)
        lse_sum = torch.zeros((q_flat.shape[0], rows), dtype=torch.float32, device=q.device)
        acc = torch.zeros((q_flat.shape[0], rows, v.shape[-1]), dtype=torch.float32, device=q.device)
        q_indices = torch.arange(q_start, q_start + rows, device=q.device)

        for k_start in range(0, nk, k_tile):
            k_chunk = k_flat[:, k_start : k_start + k_tile]
            v_chunk = v_flat[:, k_start : k_start + k_tile]
            scores = torch.einsum("bqd,bkd->bqk", q_chunk, k_chunk) * scale
            if is_causal:
                k_indices = torch.arange(k_start, k_start + k_chunk.shape[-2], device=q.device)
                mask = q_indices[:, None] >= k_indices[None, :]
                scores = scores.masked_fill(~mask, -1e6)

            scores_fp32 = scores.float()
            tile_m = scores_fp32.amax(dim=-1)
            new_m = torch.maximum(m, tile_m)
            alpha = torch.exp(m - new_m)
            probs = torch.exp(scores_fp32 - new_m.unsqueeze(-1))
            lse_sum = alpha * lse_sum + probs.sum(dim=-1)
            acc = alpha.unsqueeze(-1) * acc + torch.einsum("bqk,bkd->bqd", probs, v_chunk.float())
            m = new_m

        outputs.append((acc / lse_sum.unsqueeze(-1)).to(dtype=q.dtype))
        lses.append(m + torch.log(lse_sum))

    return torch.cat(outputs, dim=-2).reshape(*q.shape[:-2], nq, v.shape[-1]), torch.cat(lses, dim=-1).reshape(*q.shape[:-2], nq)


class FlashAttentionPyTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        out, lse = flash_attention_torch(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out):
        q, k, v, out, lse = ctx.saved_tensors
        with torch.enable_grad():
            q_ref = q.detach().requires_grad_(True)
            k_ref = k.detach().requires_grad_(True)
            v_ref = v.detach().requires_grad_(True)
            out_ref, _ = flash_attention_torch(q_ref, k_ref, v_ref, ctx.is_causal)
            dq, dk, dv = torch.autograd.grad(
                out_ref,
                (q_ref, k_ref, v_ref),
                grad_out,
                allow_unused=False,
            )
        return dq, dk, dv, None


class FlashAttentionTritonFunction(FlashAttentionPyTorchFunction):
    """FlashAttention forward path backed by a tiled Triton kernel on CUDA."""

    @staticmethod
    def forward(ctx, q, k, v, is_causal=False):
        if not q.is_cuda or triton is None:
            return FlashAttentionPyTorchFunction.forward(ctx, q, k, v, is_causal)
        out, lse = flash_attention_triton(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb,
        stride_qq,
        stride_qd,
        stride_kb,
        stride_kk,
        stride_kd,
        stride_vb,
        stride_vk,
        stride_vd,
        stride_ob,
        stride_oq,
        stride_od,
        stride_lb,
        stride_lq,
        n_queries,
        n_keys,
        scale,
        D: tl.constexpr,
        Q_TILE_SIZE: tl.constexpr,
        K_TILE_SIZE: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        query_tile = tl.program_id(0)
        batch = tl.program_id(1)
        q_offsets = query_tile * Q_TILE_SIZE + tl.arange(0, Q_TILE_SIZE)
        d_offsets = tl.arange(0, D)
        q_mask = q_offsets[:, None] < n_queries
        q = tl.load(
            q_ptr + batch * stride_qb + q_offsets[:, None] * stride_qq + d_offsets[None, :] * stride_qd,
            mask=q_mask,
            other=0.0,
        )
        acc = tl.zeros((Q_TILE_SIZE, D), dtype=tl.float32)
        m = tl.full((Q_TILE_SIZE,), -float("inf"), dtype=tl.float32)
        denom = tl.zeros((Q_TILE_SIZE,), dtype=tl.float32)

        for key_start in range(0, n_keys, K_TILE_SIZE):
            k_offsets = key_start + tl.arange(0, K_TILE_SIZE)
            k_mask = k_offsets[:, None] < n_keys
            k = tl.load(
                k_ptr + batch * stride_kb + k_offsets[:, None] * stride_kk + d_offsets[None, :] * stride_kd,
                mask=k_mask,
                other=0.0,
            )
            v = tl.load(
                v_ptr + batch * stride_vb + k_offsets[:, None] * stride_vk + d_offsets[None, :] * stride_vd,
                mask=k_mask,
                other=0.0,
            )
            scores = tl.dot(q, tl.trans(k), out_dtype=tl.float32) * scale
            valid = k_offsets[None, :] < n_keys
            if IS_CAUSAL:
                valid = valid & (q_offsets[:, None] >= k_offsets[None, :])
            scores = tl.where(valid, scores, -1e6)
            tile_m = tl.max(scores, axis=1)
            new_m = tl.maximum(m, tile_m)
            alpha = tl.exp(m - new_m)
            probs = tl.exp(scores - new_m[:, None])
            denom = alpha * denom + tl.sum(probs, axis=1)
            acc = alpha[:, None] * acc + tl.dot(probs.to(v.dtype), v, out_dtype=tl.float32)
            m = new_m

        output = acc / denom[:, None]
        tl.store(
            o_ptr + batch * stride_ob + q_offsets[:, None] * stride_oq + d_offsets[None, :] * stride_od,
            output,
            mask=q_mask,
        )
        tl.store(l_ptr + batch * stride_lb + q_offsets * stride_lq, m + tl.log(denom), mask=q_offsets < n_queries)


def flash_attention_triton(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
    if not q.is_cuda or triton is None:
        return flash_attention_torch(q, k, v, is_causal)
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Triton FlashAttention expects tensors with shape (batch, sequence, dimension)")
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    batch, n_queries, d = q.shape
    n_keys = k.shape[1]
    # The 64 x 64 tile exceeds the 4090/3090 shared-memory limit at D=128.
    q_tile = 32 if n_queries >= 32 else triton.next_power_of_2(n_queries)
    k_tile = 32 if n_keys >= 32 else triton.next_power_of_2(n_keys)
    out = torch.empty((batch, n_queries, v.shape[-1]), device=q.device, dtype=q.dtype)
    lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n_queries, q_tile), batch)
    _flash_forward_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *lse.stride(),
        n_queries,
        n_keys,
        1.0 / math.sqrt(d),
        D=d,
        Q_TILE_SIZE=q_tile,
        K_TILE_SIZE=k_tile,
        IS_CAUSAL=is_causal,
        num_warps=4,
        num_stages=1,
    )
    return out, lse


__all__ = [
    "explicit_attention",
    "FlashAttentionPyTorchFunction",
    "FlashAttentionTritonFunction",
    "flash_attention_triton",
    "flash_attention_torch",
]
