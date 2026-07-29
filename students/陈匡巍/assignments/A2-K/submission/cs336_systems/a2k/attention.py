"""Explicit attention and independently implemented FlashAttention-2 paths.

The Triton forward kernel follows the online-softmax recurrence.  The required
backward uses the saved log-sum-exp to recompute probabilities rather than
saving a quadratic attention matrix in the forward pass.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl


def _validate_inputs(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("expected q, k, v with shape [batch, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape != v.shape:
        raise ValueError("batch dimensions must match and k/v shapes must be equal")
    if q.shape[-1] != k.shape[-1]:
        raise ValueError("q and k head dimensions must be equal")
    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, v must have the same dtype")


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Attention written as QK^T, mask, softmax, and PV without fused APIs."""
    _validate_inputs(q, k, v)
    scores = torch.matmul(q, k.transpose(-2, -1)) * (q.shape[-1] ** -0.5)
    if is_causal:
        q_index = torch.arange(q.shape[-2], device=q.device)[:, None]
        k_index = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(k_index > q_index, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)


def _tiled_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool,
    query_tile: int = 32,
    key_tile: int = 32,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch online softmax used as a transparent reference."""
    _validate_inputs(q, k, v)
    batch, n_queries, head_dim = q.shape
    n_keys = k.shape[-2]
    scale = head_dim**-0.5
    output = torch.empty_like(q)
    logsumexp = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)

    for query_start in range(0, n_queries, query_tile):
        query_end = min(query_start + query_tile, n_queries)
        query = q[:, query_start:query_end].float()
        rows = query_end - query_start
        running_max = torch.full((batch, rows), -torch.inf, device=q.device, dtype=torch.float32)
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros((batch, rows, head_dim), device=q.device, dtype=torch.float32)

        for key_start in range(0, n_keys, key_tile):
            key_end = min(key_start + key_tile, n_keys)
            key = k[:, key_start:key_end].float()
            value = v[:, key_start:key_end].float()
            scores = torch.matmul(query, key.transpose(-2, -1)) * scale
            if is_causal:
                query_indices = torch.arange(query_start, query_end, device=q.device)[:, None]
                key_indices = torch.arange(key_start, key_end, device=q.device)[None, :]
                scores = scores.masked_fill(key_indices > query_indices, float("-inf"))

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, tile_max)
            old_correction = torch.exp(running_max - new_max)
            probabilities = torch.exp(scores - new_max.unsqueeze(-1))
            accumulator = accumulator * old_correction.unsqueeze(-1) + torch.matmul(probabilities, value)
            running_sum = running_sum * old_correction + probabilities.sum(dim=-1)
            running_max = new_max

        output[:, query_start:query_end] = (accumulator / running_sum.unsqueeze(-1)).to(q.dtype)
        logsumexp[:, query_start:query_end] = running_max + torch.log(running_sum)

    return output, logsumexp


def _recomputed_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Equations 13--19, with probabilities reconstructed from the saved LSE."""
    scale = q.shape[-1] ** -0.5
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        q_index = torch.arange(q.shape[-2], device=q.device)[:, None]
        k_index = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(k_index > q_index, float("-inf"))

    probabilities = torch.exp(scores.float() - logsumexp.unsqueeze(-1))
    delta = (output.float() * grad_output.float()).sum(dim=-1, keepdim=True)
    grad_v = torch.matmul(probabilities.transpose(-2, -1), grad_output.float())
    grad_probabilities = torch.matmul(grad_output.float(), v.float().transpose(-2, -1))
    grad_scores = probabilities * (grad_probabilities - delta)
    grad_q = torch.matmul(grad_scores, k.float()) * scale
    grad_k = torch.matmul(grad_scores.transpose(-2, -1), q.float()) * scale
    return grad_q.to(q.dtype), grad_k.to(k.dtype), grad_v.to(v.dtype)


def _backward_dispatch(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    grad_output: torch.Tensor,
    logsumexp: torch.Tensor,
    is_causal: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    # The compiled CUDA path fuses the elementwise recomputation.  CPU remains
    # eager so the reference test does not pay compilation overhead.
    if q.is_cuda:
        return _compiled_recomputed_backward(q, k, v, output, grad_output, logsumexp, is_causal)
    return _recomputed_backward(q, k, v, output, grad_output, logsumexp, is_causal)


_compiled_recomputed_backward = torch.compile(_recomputed_backward, backend="inductor", fullgraph=True)


class FlashAttentionPyTorch(torch.autograd.Function):
    """Tiled online-softmax reference implemented only with PyTorch."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, logsumexp = _tiled_forward(q, k, v, bool(is_causal))
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _backward_dispatch(
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
def _flash_forward_kernel(
    q_ptr,
    k_ptr,
    v_ptr,
    output_ptr,
    logsumexp_ptr,
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
    HEAD_DIM: tl.constexpr,
    QUERY_TILE: tl.constexpr,
    KEY_TILE: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
):
    query_tile_index = tl.program_id(0)
    batch_index = tl.program_id(1)
    query_start = query_tile_index * QUERY_TILE

    q_block = tl.make_block_ptr(
        q_ptr + batch_index * stride_qb,
        shape=(n_queries, HEAD_DIM),
        strides=(stride_qq, stride_qd),
        offsets=(query_start, 0),
        block_shape=(QUERY_TILE, HEAD_DIM),
        order=(1, 0),
    )
    output_block = tl.make_block_ptr(
        output_ptr + batch_index * stride_ob,
        shape=(n_queries, HEAD_DIM),
        strides=(stride_oq, stride_od),
        offsets=(query_start, 0),
        block_shape=(QUERY_TILE, HEAD_DIM),
        order=(1, 0),
    )

    query = tl.load(q_block, boundary_check=(0, 1), padding_option="zero")
    running_max = tl.full((QUERY_TILE,), -float("inf"), tl.float32)
    running_sum = tl.zeros((QUERY_TILE,), tl.float32)
    accumulator = tl.zeros((QUERY_TILE, HEAD_DIM), tl.float32)
    query_indices = query_start + tl.arange(0, QUERY_TILE)

    for key_start in range(0, n_keys, KEY_TILE):
        k_block = tl.make_block_ptr(
            k_ptr + batch_index * stride_kb,
            shape=(n_keys, HEAD_DIM),
            strides=(stride_kk, stride_kd),
            offsets=(key_start, 0),
            block_shape=(KEY_TILE, HEAD_DIM),
            order=(1, 0),
        )
        v_block = tl.make_block_ptr(
            v_ptr + batch_index * stride_vb,
            shape=(n_keys, HEAD_DIM),
            strides=(stride_vk, stride_vd),
            offsets=(key_start, 0),
            block_shape=(KEY_TILE, HEAD_DIM),
            order=(1, 0),
        )
        key = tl.load(k_block, boundary_check=(0, 1), padding_option="zero")
        value = tl.load(v_block, boundary_check=(0, 1), padding_option="zero")
        # IEEE mode matters for the required FP32 correctness cases.  Triton's
        # NVIDIA default for float32 dot products is TF32; BF16 performance
        # inputs are unaffected by this setting.
        scores = tl.dot(query, tl.trans(key), input_precision="ieee") * scale
        key_indices = key_start + tl.arange(0, KEY_TILE)
        valid = (query_indices[:, None] < n_queries) & (key_indices[None, :] < n_keys)
        if IS_CAUSAL:
            valid &= key_indices[None, :] <= query_indices[:, None]
        scores = tl.where(valid, scores, -1.0e6)

        new_max = tl.maximum(running_max, tl.max(scores, axis=1))
        old_correction = tl.exp(running_max - new_max)
        probabilities = tl.exp(scores - new_max[:, None])
        probabilities = tl.where(valid, probabilities, 0.0)
        new_sum = running_sum * old_correction + tl.sum(probabilities, axis=1)
        accumulator *= old_correction[:, None]
        accumulator = tl.dot(
            probabilities.to(value.dtype),
            value,
            acc=accumulator,
            input_precision="ieee",
        )
        running_max = new_max
        running_sum = new_sum

    normalized = accumulator / running_sum[:, None]
    tl.store(
        output_block,
        normalized.to(output_block.type.element_ty),
        boundary_check=(0, 1),
    )
    logsumexp_indices = batch_index * stride_lb + query_indices * stride_lq
    tl.store(
        logsumexp_ptr + logsumexp_indices,
        running_max + tl.log(running_sum),
        mask=query_indices < n_queries,
    )


def _kernel_config(head_dim: int) -> tuple[int, int, int, int]:
    if head_dim <= 64:
        return 64, 64, 4, 2
    return 32, 64, 4, 2


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
        _validate_inputs(q, k, v)
        if not q.is_cuda:
            raise ValueError("the Triton path requires CUDA tensors")
        if not q.is_contiguous() or not k.is_contiguous() or not v.is_contiguous():
            q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
        head_dim = q.shape[-1]
        if head_dim not in (16, 32, 64, 128):
            raise ValueError("supported head dimensions are 16, 32, 64, and 128")

        batch, n_queries, _ = q.shape
        n_keys = k.shape[-2]
        output = torch.empty_like(q)
        logsumexp = torch.empty((batch, n_queries), device=q.device, dtype=torch.float32)
        query_tile, key_tile, num_warps, num_stages = _kernel_config(head_dim)
        grid = (triton.cdiv(n_queries, query_tile), batch)
        _flash_forward_kernel[grid](
            q,
            k,
            v,
            output,
            logsumexp,
            *q.stride(),
            *k.stride(),
            *v.stride(),
            *output.stride(),
            *logsumexp.stride(),
            n_queries,
            n_keys,
            1.0 / math.sqrt(head_dim),
            HEAD_DIM=head_dim,
            QUERY_TILE=query_tile,
            KEY_TILE=key_tile,
            IS_CAUSAL=bool(is_causal),
            num_warps=num_warps,
            num_stages=num_stages,
        )
        ctx.save_for_backward(q, k, v, output, logsumexp)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, logsumexp = ctx.saved_tensors
        grad_q, grad_k, grad_v = _backward_dispatch(
            q,
            k,
            v,
            output,
            grad_output.contiguous(),
            logsumexp,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
