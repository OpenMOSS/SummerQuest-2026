"""Attention implementations used by the A2-K correctness and benchmark suites."""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Compute unfused scaled dot-product attention with explicit PyTorch ops."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        n_queries, n_keys = scores.shape[-2:]
        query_index = torch.arange(n_queries, device=scores.device)
        key_index = torch.arange(n_keys, device=scores.device)
        causal_mask = query_index[:, None] >= key_index[None, :]
        scores = scores.masked_fill(~causal_mask, -1.0e6)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def _tiled_flash_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    query_tile_size: int = 64,
    key_tile_size: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference FlashAttention-2 forward using tiled PyTorch operations."""
    batch_size, n_queries, head_dim = q.shape
    n_keys = k.shape[-2]
    scale = head_dim**-0.5
    output = torch.empty_like(q)
    logsumexp = torch.empty((batch_size, n_queries), dtype=torch.float32, device=q.device)

    for query_start in range(0, n_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, n_queries)
        q_tile = q[:, query_start:query_end].float()
        rows = query_end - query_start
        running_max = torch.full((batch_size, rows), -torch.inf, dtype=torch.float32, device=q.device)
        running_sum = torch.zeros((batch_size, rows), dtype=torch.float32, device=q.device)
        accumulator = torch.zeros((batch_size, rows, head_dim), dtype=torch.float32, device=q.device)

        for key_start in range(0, n_keys, key_tile_size):
            key_end = min(key_start + key_tile_size, n_keys)
            k_tile = k[:, key_start:key_end].float()
            v_tile = v[:, key_start:key_end].float()
            scores = torch.matmul(q_tile, k_tile.transpose(-2, -1)) * scale

            if is_causal:
                query_index = torch.arange(query_start, query_end, device=q.device)
                key_index = torch.arange(key_start, key_end, device=q.device)
                scores = scores.masked_fill(query_index[:, None] < key_index[None, :], -1.0e6)

            tile_max = scores.amax(dim=-1)
            next_max = torch.maximum(running_max, tile_max)
            correction = torch.exp(running_max - next_max)
            probabilities = torch.exp(scores - next_max.unsqueeze(-1))
            next_sum = correction * running_sum + probabilities.sum(dim=-1)
            accumulator = accumulator * correction.unsqueeze(-1) + torch.matmul(probabilities, v_tile)
            running_max = next_max
            running_sum = next_sum

        output[:, query_start:query_end] = (accumulator / running_sum.unsqueeze(-1)).to(q.dtype)
        logsumexp[:, query_start:query_end] = running_max + torch.log(running_sum)

    return output, logsumexp


def _flash_backward_recompute(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute attention probabilities and apply equations 13--19."""
    scale = q.shape[-1] ** -0.5
    q_float = q.float()
    k_float = k.float()
    v_float = v.float()
    grad_output_float = grad_output.float()

    scores = torch.matmul(q_float, k_float.transpose(-2, -1)) * scale
    if is_causal:
        n_queries, n_keys = scores.shape[-2:]
        query_index = torch.arange(n_queries, device=scores.device)
        key_index = torch.arange(n_keys, device=scores.device)
        scores = scores.masked_fill(query_index[:, None] < key_index[None, :], -1.0e6)

    probabilities = torch.exp(scores - logsumexp.float().unsqueeze(-1))
    correction = torch.sum(output.float() * grad_output_float, dim=-1, keepdim=True)
    grad_probabilities = torch.matmul(grad_output_float, v_float.transpose(-2, -1))
    grad_scores = probabilities * (grad_probabilities - correction)

    grad_q = torch.matmul(grad_scores, k_float) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q_float) * scale
    grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output_float)
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


_compiled_flash_backward = torch.compile(_flash_backward_recompute, fullgraph=True)


def _run_flash_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    backward = _compiled_flash_backward if q.is_cuda else _flash_backward_recompute
    return backward(q, k, v, output, grad_output, logsumexp, is_causal)


class FlashAttentionPyTorch(torch.autograd.Function):
    """Tiled PyTorch reference with a recomputation-based backward pass."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        output, logsumexp = _tiled_flash_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        logsumexp, q, k, v, output = ctx.saved_tensors
        grad_q, grad_k, grad_v = _run_flash_backward(
            q,
            k,
            v,
            output,
            grad_output.contiguous(),
            logsumexp,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None


@triton.jit
def _flash_attention_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    logsumexp_ptr,
    stride_q_batch,
    stride_q_sequence,
    stride_q_dim,
    stride_k_batch,
    stride_k_sequence,
    stride_k_dim,
    stride_v_batch,
    stride_v_sequence,
    stride_v_dim,
    stride_o_batch,
    stride_o_sequence,
    stride_o_dim,
    stride_l_batch,
    stride_l_sequence,
    n_queries: tl.constexpr,
    n_keys: tl.constexpr,
    head_dim: tl.constexpr,
    scale: tl.constexpr,
    is_causal: tl.constexpr,
    query_tile_size: tl.constexpr,
    key_tile_size: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_offsets = query_tile_index * query_tile_size + tl.arange(0, query_tile_size)
    dim_offsets = tl.arange(0, head_dim)
    query_mask = query_offsets < n_queries

    q_offsets = (
        batch_index * stride_q_batch
        + query_offsets[:, None] * stride_q_sequence
        + dim_offsets[None, :] * stride_q_dim
    )
    q = tl.load(q_ptr + q_offsets, mask=query_mask[:, None], other=0.0)

    running_max = tl.full((query_tile_size,), -float("inf"), tl.float32)
    running_sum = tl.zeros((query_tile_size,), tl.float32)
    accumulator = tl.zeros((query_tile_size, head_dim), tl.float32)

    for key_start in tl.range(0, n_keys, key_tile_size):
        key_offsets = key_start + tl.arange(0, key_tile_size)
        key_mask = key_offsets < n_keys
        k_offsets = (
            batch_index * stride_k_batch
            + key_offsets[None, :] * stride_k_sequence
            + dim_offsets[:, None] * stride_k_dim
        )
        v_offsets = (
            batch_index * stride_v_batch
            + key_offsets[:, None] * stride_v_sequence
            + dim_offsets[None, :] * stride_v_dim
        )
        k = tl.load(k_ptr + k_offsets, mask=key_mask[None, :], other=0.0)
        v = tl.load(v_ptr + v_offsets, mask=key_mask[:, None], other=0.0)

        scores = tl.dot(q, k) * scale
        valid_scores = query_mask[:, None] & key_mask[None, :]
        if is_causal:
            valid_scores = valid_scores & (query_offsets[:, None] >= key_offsets[None, :])
        scores = tl.where(valid_scores, scores, -1.0e6)

        tile_max = tl.max(scores, axis=1)
        next_max = tl.maximum(running_max, tile_max)
        correction = tl.exp(running_max - next_max)
        probabilities = tl.exp(scores - next_max[:, None])
        next_sum = correction * running_sum + tl.sum(probabilities, axis=1)
        accumulator = accumulator * correction[:, None] + tl.dot(probabilities.to(v.dtype), v)
        running_max = next_max
        running_sum = next_sum

    normalized_output = accumulator / running_sum[:, None]
    output_offsets = (
        batch_index * stride_o_batch
        + query_offsets[:, None] * stride_o_sequence
        + dim_offsets[None, :] * stride_o_dim
    )
    logsumexp_offsets = batch_index * stride_l_batch + query_offsets * stride_l_sequence
    tl.store(output_ptr + output_offsets, normalized_output, mask=query_mask[:, None])
    tl.store(logsumexp_ptr + logsumexp_offsets, running_max + tl.log(running_sum), mask=query_mask)


class FlashAttentionTriton(torch.autograd.Function):
    """FlashAttention-2 with a student-written Triton forward kernel."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        if not q.is_cuda:
            raise RuntimeError("FlashAttentionTriton requires a CUDA tensor")
        if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
            raise ValueError("expected q, k, and v with shape [batch, sequence, head_dim]")
        if q.shape[0] != k.shape[0] or k.shape != v.shape or q.shape[-1] != k.shape[-1]:
            raise ValueError("incompatible q, k, and v shapes")
        if q.shape[-1] not in {16, 32, 64, 128}:
            raise ValueError("head_dim must be one of 16, 32, 64, or 128")

        q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        batch_size, n_queries, head_dim = q.shape
        n_keys = k.shape[-2]
        output = torch.empty_like(q)
        logsumexp = torch.empty((batch_size, n_queries), dtype=torch.float32, device=q.device)
        query_tile_size = 64
        key_tile_size = 64 if head_dim <= 64 else 32
        grid = (triton.cdiv(n_queries, query_tile_size), batch_size)

        _flash_attention_forward_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            q.stride(0),
            q.stride(1),
            q.stride(2),
            k.stride(0),
            k.stride(1),
            k.stride(2),
            v.stride(0),
            v.stride(1),
            v.stride(2),
            output.stride(0),
            output.stride(1),
            output.stride(2),
            logsumexp.stride(0),
            logsumexp.stride(1),
            n_queries=n_queries,
            n_keys=n_keys,
            head_dim=head_dim,
            scale=1.0 / math.sqrt(head_dim),
            is_causal=bool(is_causal),
            query_tile_size=query_tile_size,
            key_tile_size=key_tile_size,
            num_warps=4,
            num_stages=2,
        )

        ctx.save_for_backward(logsumexp, q, k, v, output)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        logsumexp, q, k, v, output = ctx.saved_tensors
        grad_q, grad_k, grad_v = _run_flash_backward(
            q,
            k,
            v,
            output,
            grad_output.contiguous(),
            logsumexp,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
