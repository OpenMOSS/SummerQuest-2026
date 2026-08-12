"""Independent tiled-attention implementations for A2-K."""

from __future__ import annotations

import math
from typing import Any

import torch

try:
    import triton
    import triton.language as tl
except (ImportError, ModuleNotFoundError):
    triton = None
    tl = None


def _tiled_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    q_tile: int = 64,
    k_tile: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute attention without materialising the full score matrix."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k and v must have shape [batch, sequence, head_dim]")
    if (
        q.shape[0] != k.shape[0]
        or k.shape[:2] != v.shape[:2]
        or q.shape[-1] != k.shape[-1]
    ):
        raise ValueError("incompatible attention shapes")
    bsz, n_queries, head_dim = q.shape
    n_keys = k.shape[1]
    scale = 1.0 / math.sqrt(head_dim)
    output = torch.zeros(
        (bsz, n_queries, v.shape[-1]), device=q.device, dtype=torch.float32
    )
    lse = torch.empty((bsz, n_queries), device=q.device, dtype=torch.float32)
    qf, kf, vf = q.float(), k.float(), v.float()
    key_positions = torch.arange(n_keys, device=q.device)

    for q0 in range(0, n_queries, q_tile):
        q1 = min(q0 + q_tile, n_queries)
        q_chunk = qf[:, q0:q1]
        rows = q1 - q0
        running_max = torch.full((bsz, rows), -float("inf"), device=q.device)
        running_sum = torch.zeros((bsz, rows), device=q.device)
        running_out = torch.zeros((bsz, rows, v.shape[-1]), device=q.device)
        query_positions = torch.arange(q0, q1, device=q.device)

        for k0 in range(0, n_keys, k_tile):
            k1 = min(k0 + k_tile, n_keys)
            scores = torch.matmul(q_chunk, kf[:, k0:k1].transpose(-1, -2)) * scale
            if is_causal:
                allowed = query_positions[:, None] >= key_positions[k0:k1][None, :]
                scores = scores.masked_fill(~allowed[None, :, :], -1.0e9)
            block_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, block_max)
            old_scale = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            new_sum = running_sum * old_scale + probabilities.sum(dim=-1)
            running_out = running_out * old_scale.unsqueeze(-1) + torch.matmul(
                probabilities, vf[:, k0:k1]
            )
            running_max, running_sum = new_max, new_sum

        running_sum = running_sum.clamp_min(torch.finfo(running_sum.dtype).tiny)
        output[:, q0:q1] = running_out / running_sum.unsqueeze(-1)
        lse[:, q0:q1] = running_max + running_sum.log()
    return output.to(dtype=q.dtype), lse


class FlashAttentionPyTorch(torch.autograd.Function):
    """Tiled reference implementation with a recomputation backward."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ):
        out, lse = _tiled_attention(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        with torch.enable_grad():
            q1 = q.detach().requires_grad_(q.requires_grad)
            k1 = k.detach().requires_grad_(k.requires_grad)
            v1 = v.detach().requires_grad_(v.requires_grad)
            out, _ = _tiled_attention(q1, k1, v1, ctx.is_causal)
            grads = torch.autograd.grad(out, (q1, k1, v1), do, allow_unused=True)
        return (*grads, None)


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        n_queries,
        n_keys,
        head_dim: tl.constexpr,
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
        scale: tl.constexpr,
        causal: tl.constexpr,
        Q_TILE: tl.constexpr,
        K_TILE: tl.constexpr,
        D_BLOCK: tl.constexpr,
    ):
        query_block = tl.program_id(0)
        batch = tl.program_id(1)
        rows = query_block * Q_TILE + tl.arange(0, Q_TILE)
        dims = tl.arange(0, D_BLOCK)
        q_offsets = (
            batch * stride_qb + rows[:, None] * stride_qq + dims[None, :] * stride_qd
        )
        q_mask = (rows[:, None] < n_queries) & (dims[None, :] < head_dim)
        q_values = tl.load(q_ptr + q_offsets, mask=q_mask, other=0.0)
        running_max = tl.full((Q_TILE,), -float("inf"), tl.float32)
        running_sum = tl.zeros((Q_TILE,), tl.float32)
        accumulator = tl.zeros((Q_TILE, D_BLOCK), tl.float32)

        for key_start in range(0, n_keys, K_TILE):
            keys = key_start + tl.arange(0, K_TILE)
            k_offsets = (
                batch * stride_kb
                + keys[None, :] * stride_kk
                + dims[:, None] * stride_kd
            )
            k_values = tl.load(
                k_ptr + k_offsets,
                mask=(keys[None, :] < n_keys) & (dims[:, None] < head_dim),
                other=0.0,
            )
            scores = tl.dot(q_values, k_values) * scale
            score_mask = (rows[:, None] < n_queries) & (keys[None, :] < n_keys)
            if causal:
                score_mask = score_mask & (rows[:, None] >= keys[None, :])
            scores = tl.where(score_mask, scores, -1.0e9)
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            old_scale = tl.exp(running_max - new_max)
            probabilities = tl.exp(scores - new_max[:, None])
            v_offsets = (
                batch * stride_vb
                + keys[:, None] * stride_vk
                + dims[None, :] * stride_vd
            )
            v_values = tl.load(
                v_ptr + v_offsets,
                mask=(keys[:, None] < n_keys) & (dims[None, :] < head_dim),
                other=0.0,
            )
            v_values = v_values.to(tl.float32)
            accumulator = accumulator * old_scale[:, None] + tl.dot(
                probabilities, v_values
            )
            running_sum = running_sum * old_scale + tl.sum(probabilities, axis=1)
            running_max = new_max

        output = accumulator / running_sum[:, None]
        o_offsets = (
            batch * stride_ob + rows[:, None] * stride_oq + dims[None, :] * stride_od
        )
        tl.store(o_ptr + o_offsets, output, mask=q_mask)
        tl.store(
            l_ptr + batch * stride_lb + rows * stride_lq,
            running_max + tl.log(running_sum),
            mask=rows < n_queries,
        )

else:
    _flash_forward_kernel = None


def _triton_forward(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool):
    if triton is None or _flash_forward_kernel is None:
        raise RuntimeError("Triton is not installed")
    if not q.is_cuda:
        raise RuntimeError("Triton FlashAttention requires CUDA tensors")
    if (
        q.ndim != 3
        or k.ndim != 3
        or v.ndim != 3
        or q.shape[0] != k.shape[0]
        or k.shape[:2] != v.shape[:2]
        or q.shape[-1] != k.shape[-1]
        or q.shape[-1] != v.shape[-1]
    ):
        raise ValueError("incompatible attention shapes")
    bsz, n_queries, dim = q.shape
    n_keys = k.shape[1]
    d_block = triton.next_power_of_2(dim)
    out = torch.empty_like(q)
    lse = torch.empty((bsz, n_queries), device=q.device, dtype=torch.float32)
    grid = (triton.cdiv(n_queries, 64), bsz)
    _flash_forward_kernel[grid](
        q,
        k,
        v,
        out,
        lse,
        n_queries,
        n_keys,
        dim,
        *q.stride(),
        *k.stride(),
        *v.stride(),
        *out.stride(),
        *lse.stride(),
        1.0 / math.sqrt(dim),
        causal=is_causal,
        Q_TILE=64,
        K_TILE=64,
        D_BLOCK=d_block,
        num_warps=4,
        num_stages=2,
    )
    return out, lse


class FlashAttentionTriton(torch.autograd.Function):
    """Triton tiled forward; backward recomputes the tiled PyTorch graph."""

    @staticmethod
    def forward(
        ctx: Any,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ):
        out, lse = _triton_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, out, lse)
        ctx.is_causal = bool(is_causal)
        return out

    @staticmethod
    def backward(ctx: Any, do: torch.Tensor):
        q, k, v, _out, _lse = ctx.saved_tensors
        with torch.enable_grad():
            q1 = q.detach().requires_grad_(q.requires_grad)
            k1 = k.detach().requires_grad_(k.requires_grad)
            v1 = v.detach().requires_grad_(v.requires_grad)
            out, _ = _tiled_attention(q1, k1, v1, ctx.is_causal)
            grads = torch.autograd.grad(out, (q1, k1, v1), do, allow_unused=True)
        return (*grads, None)


__all__ = ["FlashAttentionPyTorch", "FlashAttentionTriton", "_tiled_attention"]
