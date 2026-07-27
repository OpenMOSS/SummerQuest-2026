"""Tiled PyTorch and student-written Triton FlashAttention-2 forward paths."""

from __future__ import annotations

import math

import torch

try:
    import triton
    import triton.language as tl
except ImportError:  # CPU-only import remains useful for the PyTorch reference.
    triton = None
    tl = None


def _check_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("Q, K, and V must have shape [batch, sequence, head_dim]")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("self-attention requires Q, K, and V with identical shapes")
    if q.shape[-1] not in (16, 32, 64, 128):
        raise ValueError("head_dim must be one of 16, 32, 64, or 128")


def _mask_scores(scores: torch.Tensor, q_start: int, k_start: int, causal: bool) -> torch.Tensor:
    if not causal:
        return scores
    q_pos = torch.arange(q_start, q_start + scores.shape[-2], device=scores.device)[:, None]
    k_pos = torch.arange(k_start, k_start + scores.shape[-1], device=scores.device)[None, :]
    return scores.masked_fill(q_pos < k_pos, float("-inf"))


def _pytorch_tiled_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    causal: bool,
    block_q: int = 64,
    block_k: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Online-softmax reference that never materializes the full score matrix."""
    _check_inputs(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    output = torch.empty_like(q)
    lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
    for q_start in range(0, q.shape[-2], block_q):
        q_end = min(q_start + block_q, q.shape[-2])
        q_block = q[:, q_start:q_end].float()
        m = torch.full(q_block.shape[:-1], -torch.inf, device=q.device)
        denominator = torch.zeros_like(m)
        accumulator = torch.zeros_like(q_block)
        for k_start in range(0, k.shape[-2], block_k):
            k_end = min(k_start + block_k, k.shape[-2])
            scores = torch.matmul(q_block, k[:, k_start:k_end].float().transpose(-1, -2)) * scale
            scores = _mask_scores(scores, q_start, k_start, causal)
            new_m = torch.maximum(m, scores.amax(dim=-1))
            alpha = torch.exp(m - new_m)
            probabilities = torch.exp(scores - new_m.unsqueeze(-1))
            denominator = denominator * alpha + probabilities.sum(dim=-1)
            accumulator = accumulator * alpha.unsqueeze(-1) + torch.matmul(probabilities, v[:, k_start:k_end].float())
            m = new_m
        output[:, q_start:q_end] = (accumulator / denominator.unsqueeze(-1)).to(q.dtype)
        lse[:, q_start:q_end] = m + torch.log(denominator)
    return output, lse


def _recomputed_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    causal: bool,
    block_q: int = 64,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute attention probabilities blockwise and apply the exact VJP."""
    scale = 1.0 / math.sqrt(q.shape[-1])
    dq = torch.zeros_like(q)
    dk = torch.zeros_like(k)
    dv = torch.zeros_like(v)
    k_t = k.float().transpose(-1, -2)
    for q_start in range(0, q.shape[-2], block_q):
        q_end = min(q_start + block_q, q.shape[-2])
        q_block = q[:, q_start:q_end]
        do_block = grad_output[:, q_start:q_end]
        scores = torch.matmul(q_block.float(), k_t) * scale
        scores = _mask_scores(scores, q_start, 0, causal)
        probabilities = torch.exp(scores - lse[:, q_start:q_end].unsqueeze(-1))
        dp = torch.matmul(do_block.float(), v.float().transpose(-1, -2))
        delta = (do_block.float() * output[:, q_start:q_end].float()).sum(dim=-1, keepdim=True)
        ds = probabilities * (dp - delta)
        dq[:, q_start:q_end] = (torch.matmul(ds, k.float()) * scale).to(q.dtype)
        dk.add_((torch.matmul(ds.transpose(-1, -2), q_block.float()) * scale).to(k.dtype))
        dv.add_(torch.matmul(probabilities.transpose(-1, -2), do_block.float()).to(v.dtype))
    return dq, dk, dv


if triton is not None:

    @triton.jit
    def _flash_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        o_ptr,
        l_ptr,
        stride_qb: tl.constexpr,
        stride_qn: tl.constexpr,
        stride_qd: tl.constexpr,
        stride_kb: tl.constexpr,
        stride_kn: tl.constexpr,
        stride_kd: tl.constexpr,
        stride_vb: tl.constexpr,
        stride_vn: tl.constexpr,
        stride_vd: tl.constexpr,
        stride_ob: tl.constexpr,
        stride_on: tl.constexpr,
        stride_od: tl.constexpr,
        stride_lb: tl.constexpr,
        stride_ln: tl.constexpr,
        n_ctx: tl.constexpr,
        scale: tl.constexpr,
        is_causal: tl.constexpr,
        block_q: tl.constexpr,
        block_k: tl.constexpr,
        head_dim: tl.constexpr,
    ):
        q_block = tl.program_id(0)
        batch = tl.program_id(1)
        q_offsets = q_block * block_q + tl.arange(0, block_q)
        d_offsets = tl.arange(0, head_dim)
        q_mask = q_offsets < n_ctx
        q = tl.load(q_ptr + batch * stride_qb + q_offsets[:, None] * stride_qn + d_offsets[None, :] * stride_qd, mask=q_mask[:, None], other=0.0)
        running_max = tl.full((block_q,), -float("inf"), tl.float32)
        running_sum = tl.zeros((block_q,), tl.float32)
        accumulator = tl.zeros((block_q, head_dim), tl.float32)

        for k_start in range(0, n_ctx, block_k):
            k_offsets = k_start + tl.arange(0, block_k)
            k_mask = k_offsets < n_ctx
            k = tl.load(k_ptr + batch * stride_kb + k_offsets[:, None] * stride_kn + d_offsets[None, :] * stride_kd, mask=k_mask[:, None], other=0.0)
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * scale
            valid = q_mask[:, None] & k_mask[None, :]
            if is_causal:
                valid &= q_offsets[:, None] >= k_offsets[None, :]
            scores = tl.where(valid, scores, -float("inf"))
            block_max = tl.max(scores, axis=1)
            new_max = tl.maximum(running_max, block_max)
            alpha = tl.exp2((running_max - new_max) * 1.4426950408889634)
            probabilities = tl.exp2((scores - new_max[:, None]) * 1.4426950408889634)
            v = tl.load(v_ptr + batch * stride_vb + k_offsets[:, None] * stride_vn + d_offsets[None, :] * stride_vd, mask=k_mask[:, None], other=0.0)
            accumulator = accumulator * alpha[:, None] + tl.dot(probabilities.to(v.dtype), v, input_precision="ieee")
            running_sum = running_sum * alpha + tl.sum(probabilities, axis=1)
            running_max = new_max

        output = accumulator / running_sum[:, None]
        tl.store(o_ptr + batch * stride_ob + q_offsets[:, None] * stride_on + d_offsets[None, :] * stride_od, output, mask=q_mask[:, None])
        tl.store(l_ptr + batch * stride_lb + q_offsets * stride_ln, running_max + tl.log(running_sum), mask=q_mask)


class FlashAttentionPyTorch(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        output, lse = _pytorch_tiled_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _recomputed_backward(q, k, v, output, lse, grad_output, ctx.is_causal)
        return dq, dk, dv, None


class FlashAttentionTriton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False):
        _check_inputs(q, k, v)
        if triton is None:
            raise RuntimeError("Triton is not installed")
        if not q.is_cuda:
            raise RuntimeError("The Triton implementation requires CUDA tensors")
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output = torch.empty_like(q)
        lse = torch.empty(q.shape[:-1], device=q.device, dtype=torch.float32)
        block_q, block_k = 64, 64
        grid = (triton.cdiv(q.shape[1], block_q), q.shape[0])
        _flash_forward_kernel[grid](
            q, k, v, output, lse,
            *q.stride(), *k.stride(), *v.stride(), *output.stride(), *lse.stride(),
            n_ctx=q.shape[1], scale=1.0 / math.sqrt(q.shape[-1]), is_causal=bool(is_causal),
            block_q=block_q, block_k=block_k, head_dim=q.shape[-1], num_warps=4, num_stages=2,
        )
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        q, k, v, output, lse = ctx.saved_tensors
        dq, dk, dv = _recomputed_backward(q, k, v, output, lse, grad_output, ctx.is_causal)
        return dq, dk, dv, None


__all__ = ["FlashAttentionPyTorch", "FlashAttentionTriton"]
