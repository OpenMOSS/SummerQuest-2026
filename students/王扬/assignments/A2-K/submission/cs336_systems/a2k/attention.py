from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # pragma: no cover - exercised only in environments without Triton.
    triton = None
    tl = None


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Explicit PyTorch attention baseline: QK^T, scale, optional causal mask, softmax, PV."""
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores * (1.0 / math.sqrt(d))
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)
        k_idx = torch.arange(n_keys, device=q.device)
        causal_mask = q_idx[..., None] >= k_idx[None, ...]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    return probs @ v


def _attention_and_lse(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    scores = q @ k.transpose(-2, -1)
    scores = scores * (1.0 / math.sqrt(d))
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        q_idx = torch.arange(n_queries, device=q.device)
        k_idx = torch.arange(n_keys, device=q.device)
        causal_mask = q_idx[..., None] >= k_idx[None, ...]
        scores = scores.masked_fill(~causal_mask, torch.finfo(scores.dtype).min)
    probs = torch.softmax(scores, dim=-1)
    out = probs @ v
    lse = torch.logsumexp(scores, dim=-1)
    return out, lse


def _tiled_attention_and_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    query_tile_size: int = 32,
    key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    d = q.shape[-1]
    n_queries = q.shape[-2]
    n_keys = k.shape[-2]
    out_tiles = []
    lse_tiles = []
    scale = 1.0 / math.sqrt(d)

    for q_start in range(0, n_queries, query_tile_size):
        q_end = min(q_start + query_tile_size, n_queries)
        q_tile = q[..., q_start:q_end, :]
        m_i = torch.full(q_tile.shape[:-1], -torch.inf, device=q.device, dtype=torch.float32)
        l_i = torch.zeros(q_tile.shape[:-1], device=q.device, dtype=torch.float32)
        acc = torch.zeros_like(q_tile, dtype=torch.float32)

        for k_start in range(0, n_keys, key_tile_size):
            k_end = min(k_start + key_tile_size, n_keys)
            k_tile = k[..., k_start:k_end, :]
            v_tile = v[..., k_start:k_end, :]
            scores = (q_tile.float() @ k_tile.float().transpose(-2, -1)) * scale
            if is_causal:
                q_idx = torch.arange(q_start, q_end, device=q.device)
                k_idx = torch.arange(k_start, k_end, device=q.device)
                scores = scores.masked_fill(~(q_idx[:, None] >= k_idx[None, :]), -torch.inf)

            m_new = torch.maximum(m_i, scores.max(dim=-1).values)
            p = torch.exp(scores - m_new[..., None])
            alpha = torch.exp(m_i - m_new)
            l_new = l_i * alpha + p.sum(dim=-1)
            acc = acc * alpha[..., None] + p.to(torch.float32) @ v_tile.float()
            m_i = m_new
            l_i = l_new

        out_tiles.append((acc / l_i[..., None]).to(q.dtype))
        lse_tiles.append(m_i + torch.log(l_i))

    return torch.cat(out_tiles, dim=-2), torch.cat(lse_tiles, dim=-1)


def _recompute_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_out: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        q_detached = q.detach().requires_grad_(True)
        k_detached = k.detach().requires_grad_(True)
        v_detached = v.detach().requires_grad_(True)
        out, _ = _attention_and_lse(q_detached, k_detached, v_detached, is_causal)
        dq, dk, dv = torch.autograd.grad(out, (q_detached, k_detached, v_detached), grad_out)
    return dq, dk, dv


class FlashAttentionPyTorchFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        out, lse = _tiled_attention_and_lse(q, k, v, is_causal)
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(q, k, v, grad_out, ctx.is_causal)
        return dq, dk, dv, None


if triton is not None:

    @triton.jit
    def _flash_fwd_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        lse_ptr,
        stride_qb: tl.constexpr,
        stride_qq: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kk: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vk: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_oq: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_lq: tl.constexpr,
        n_queries: tl.constexpr,
        n_keys: tl.constexpr,
        head_dim: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        pid_m = tl.program_id(0)
        pid_b = tl.program_id(1)

        offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        offs_d = tl.arange(0, BLOCK_D)

        q = tl.load(
            q_ptr + pid_b * stride_qb + offs_m[:, None] * stride_qq + offs_d[None, :] * stride_qd,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < head_dim),
            other=0.0,
        )

        m_i = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        l_i = tl.zeros((BLOCK_M,), tl.float32)
        acc = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for start_n in range(0, n_keys, BLOCK_N):
            cur_n = start_n + offs_n
            k = tl.load(
                k_ptr + pid_b * stride_kb + cur_n[None, :] * stride_kk + offs_d[:, None] * stride_kd,
                mask=(cur_n[None, :] < n_keys) & (offs_d[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q, k, input_precision="ieee").to(tl.float32) * scale
            valid = (offs_m[:, None] < n_queries) & (cur_n[None, :] < n_keys)
            if is_causal:
                valid = valid & (offs_m[:, None] >= cur_n[None, :])
            scores = tl.where(valid, scores, -float("inf"))

            m_ij = tl.maximum(m_i, tl.max(scores, axis=1))
            p = tl.exp(scores - m_ij[:, None])
            alpha = tl.exp(m_i - m_ij)
            l_ij = l_i * alpha + tl.sum(p, axis=1)

            v_tile = tl.load(
                v_ptr + pid_b * stride_vb + cur_n[:, None] * stride_vk + offs_d[None, :] * stride_vd,
                mask=(cur_n[:, None] < n_keys) & (offs_d[None, :] < head_dim),
                other=0.0,
            )
            acc = acc * alpha[:, None] + tl.dot(p.to(v_tile.dtype), v_tile, input_precision="ieee")
            m_i = m_ij
            l_i = l_ij

        out = acc / l_i[:, None]
        tl.store(
            o_ptr + pid_b * stride_ob + offs_m[:, None] * stride_oq + offs_d[None, :] * stride_od,
            out,
            mask=(offs_m[:, None] < n_queries) & (offs_d[None, :] < head_dim),
        )
        tl.store(lse_ptr + pid_b * stride_lb + offs_m * stride_lq, m_i + tl.log(l_i), mask=offs_m < n_queries)


def _triton_block_sizes(head_dim: int) -> tuple[int, int, int, int, int]:
    block_m = 16
    block_n = 32
    block_d = triton.next_power_of_2(head_dim) if triton is not None else head_dim
    num_warps = 4
    num_stages = 3
    return block_m, block_n, block_d, num_warps, num_stages


class FlashAttentionTritonFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
        if not q.is_cuda:
            out, lse = _attention_and_lse(q, k, v, is_causal)
        else:
            if triton is None:
                raise RuntimeError("Triton is required for CUDA FlashAttention forward.")
            if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
                raise ValueError("Expected q, k, v with shape [batch, sequence, head_dim].")
            if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
                raise ValueError("Expected matching batch size, key/value shape, and head dimension.")

            q_c = q.contiguous()
            k_c = k.contiguous()
            v_c = v.contiguous()
            batch, n_queries, head_dim = q_c.shape
            n_keys = k_c.shape[1]
            out = torch.empty_like(q_c)
            lse = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
            block_m, block_n, block_d, num_warps, num_stages = _triton_block_sizes(head_dim)
            grid = (triton.cdiv(n_queries, block_m), batch)
            _flash_fwd_kernel[grid](
                q_c,
                k_c,
                v_c,
                out,
                lse,
                q_c.stride(0),
                q_c.stride(1),
                q_c.stride(2),
                k_c.stride(0),
                k_c.stride(1),
                k_c.stride(2),
                v_c.stride(0),
                v_c.stride(1),
                v_c.stride(2),
                out.stride(0),
                out.stride(1),
                out.stride(2),
                lse.stride(0),
                lse.stride(1),
                n_queries,
                n_keys,
                head_dim,
                1.0 / math.sqrt(head_dim),
                bool(is_causal),
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                BLOCK_D=block_d,
                num_warps=num_warps,
                num_stages=num_stages,
            )
            q, k, v = q_c, k_c, v_c

        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx, grad_out: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        dq, dk, dv = _recompute_backward(q, k, v, grad_out, ctx.is_causal)
        return dq, dk, dv, None
