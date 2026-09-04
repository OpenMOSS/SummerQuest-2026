"""Student-written Triton FlashAttention-2 forward kernel.

The forward path is a real ``@triton.jit`` kernel with one program per query
tile and a loop over key/value tiles.  The required backward path uses the
same PyTorch recomputation routine as the tiled reference, which is explicitly
permitted by the A2-K handout.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor

from .attention import validate_attention_inputs
from .flash_pytorch import _recompute_gradients

try:
    import triton
    import triton.language as tl
except ModuleNotFoundError:  # Keep CPU-only imports usable; CUDA invocation fails explicitly below.
    triton = None  # type: ignore[assignment]
    tl = None  # type: ignore[assignment]


if triton is not None:

    @triton.jit
    def _flash_attention_forward_kernel(
        q_ptr,
        k_ptr,
        v_ptr,
        output_ptr,
        lse_ptr,
        stride_q_batch,
        stride_q_sequence,
        stride_q_dim,
        stride_k_batch,
        stride_k_sequence,
        stride_k_dim,
        stride_v_batch,
        stride_v_sequence,
        stride_v_dim,
        stride_output_batch,
        stride_output_sequence,
        stride_output_dim,
        stride_lse_batch,
        stride_lse_sequence,
        N_QUERIES: tl.constexpr,
        N_KEYS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        SCALE: tl.constexpr,
        BLOCK_M: tl.constexpr,
        BLOCK_N: tl.constexpr,
        BLOCK_D: tl.constexpr,
        IS_CAUSAL: tl.constexpr,
    ):
        program_query_tile = tl.program_id(0)
        batch_index = tl.program_id(1)
        query_offsets = program_query_tile * BLOCK_M + tl.arange(0, BLOCK_M)
        dimension_offsets = tl.arange(0, BLOCK_D)

        q_offsets = (
            batch_index * stride_q_batch
            + query_offsets[:, None] * stride_q_sequence
            + dimension_offsets[None, :] * stride_q_dim
        )
        q = tl.load(
            q_ptr + q_offsets,
            mask=(query_offsets[:, None] < N_QUERIES) & (dimension_offsets[None, :] < HEAD_DIM),
            other=0.0,
        )

        row_max = tl.full((BLOCK_M,), -float("inf"), tl.float32)
        row_sum = tl.zeros((BLOCK_M,), tl.float32)
        accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)

        for key_start in range(0, N_KEYS, BLOCK_N):
            key_offsets = key_start + tl.arange(0, BLOCK_N)
            k_offsets = (
                batch_index * stride_k_batch
                + key_offsets[:, None] * stride_k_sequence
                + dimension_offsets[None, :] * stride_k_dim
            )
            k = tl.load(
                k_ptr + k_offsets,
                mask=(key_offsets[:, None] < N_KEYS) & (dimension_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )
            # FP32 correctness runs must not silently use TF32 Tensor Cores.
            # This setting is ignored for the required BF16 performance path.
            scores = tl.dot(q, tl.trans(k), input_precision="ieee") * SCALE
            valid = (query_offsets[:, None] < N_QUERIES) & (key_offsets[None, :] < N_KEYS)
            if IS_CAUSAL:
                valid = valid & (query_offsets[:, None] >= key_offsets[None, :])
            scores = tl.where(valid, scores, -float("inf"))

            next_row_max = tl.maximum(row_max, tl.max(scores, axis=1))
            rescale_previous = tl.exp(row_max - next_row_max)
            probabilities = tl.exp(scores - next_row_max[:, None])
            next_row_sum = row_sum * rescale_previous + tl.sum(probabilities, axis=1)

            v_offsets = (
                batch_index * stride_v_batch
                + key_offsets[:, None] * stride_v_sequence
                + dimension_offsets[None, :] * stride_v_dim
            )
            value = tl.load(
                v_ptr + v_offsets,
                mask=(key_offsets[:, None] < N_KEYS) & (dimension_offsets[None, :] < HEAD_DIM),
                other=0.0,
            )
            accumulator = accumulator * rescale_previous[:, None] + tl.dot(
                probabilities.to(value.dtype), value, input_precision="ieee"
            )
            row_max = next_row_max
            row_sum = next_row_sum

        output_offsets = (
            batch_index * stride_output_batch
            + query_offsets[:, None] * stride_output_sequence
            + dimension_offsets[None, :] * stride_output_dim
        )
        output = accumulator / row_sum[:, None]
        tl.store(
            output_ptr + output_offsets,
            output,
            mask=(query_offsets[:, None] < N_QUERIES) & (dimension_offsets[None, :] < HEAD_DIM),
        )
        lse_offsets = batch_index * stride_lse_batch + query_offsets * stride_lse_sequence
        tl.store(lse_ptr + lse_offsets, row_max + tl.log(row_sum), mask=query_offsets < N_QUERIES)

else:
    _flash_attention_forward_kernel: Any = None


def triton_tile_configuration(head_dim: int) -> dict[str, int]:
    """Return the fixed, auditable kernel launch configuration for ``head_dim``.

    A single pipeline stage keeps the FP32 head_dim=128 specialization below
    the 4090 shared-memory limit. The former two-stage variant required 115712
    bytes while the device limit is 101376 bytes.
    """

    if head_dim <= 0 or head_dim > 128:
        raise ValueError("the A2-K Triton kernel supports head dimensions in [1, 128]")
    if triton is None:
        raise RuntimeError("Triton is not installed in this Python environment")
    return {
        "BLOCK_M": 64,
        "BLOCK_N": 64,
        "BLOCK_D": triton.next_power_of_2(head_dim),
        "num_warps": 4,
        "num_stages": 1,
    }


def _launch_triton_forward(q: Tensor, k: Tensor, v: Tensor, is_causal: bool) -> tuple[Tensor, Tensor]:
    """Launch the custom Triton online-softmax forward kernel."""

    shape = validate_attention_inputs(q, k, v)
    if triton is None or _flash_attention_forward_kernel is None:
        raise RuntimeError("Triton is required to execute FlashAttentionTritonFunction")
    if not q.is_cuda:
        raise RuntimeError("FlashAttentionTritonFunction requires CUDA tensors")

    configuration = triton_tile_configuration(shape.head_dim)
    q_contiguous = q.contiguous()
    k_contiguous = k.contiguous()
    v_contiguous = v.contiguous()
    output = torch.empty_like(q_contiguous)
    lse = torch.empty((shape.batch_size, shape.n_queries), dtype=torch.float32, device=q.device)
    grid = (triton.cdiv(shape.n_queries, configuration["BLOCK_M"]), shape.batch_size)
    _flash_attention_forward_kernel[grid](
        q_contiguous,
        k_contiguous,
        v_contiguous,
        output,
        lse,
        q_contiguous.stride(0),
        q_contiguous.stride(1),
        q_contiguous.stride(2),
        k_contiguous.stride(0),
        k_contiguous.stride(1),
        k_contiguous.stride(2),
        v_contiguous.stride(0),
        v_contiguous.stride(1),
        v_contiguous.stride(2),
        output.stride(0),
        output.stride(1),
        output.stride(2),
        lse.stride(0),
        lse.stride(1),
        N_QUERIES=shape.n_queries,
        N_KEYS=shape.n_keys,
        HEAD_DIM=shape.head_dim,
        SCALE=1.0 / math.sqrt(shape.head_dim),
        IS_CAUSAL=bool(is_causal),
        **configuration,
    )
    return output, lse


class FlashAttentionTritonFunction(torch.autograd.Function):
    """Autograd bridge for the custom Triton forward and recomputed backward."""

    BENCHMARK_CONFIG = {
        "query_tile": 64,
        "key_tile": 64,
        "num_warps": 4,
        "num_stages": 1,
    }

    @staticmethod
    def forward(ctx: torch.autograd.function.FunctionCtx, q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
        output, lse = _launch_triton_forward(q, k, v, bool(is_causal))
        # Keep the same saved-tensor contract as the PyTorch implementation.
        ctx.save_for_backward(q, k, v, output, lse)
        ctx.is_causal = bool(is_causal)
        return output

    @staticmethod
    def backward(ctx: torch.autograd.function.FunctionCtx, grad_output: Tensor) -> tuple[Tensor | None, Tensor | None, Tensor | None, None]:
        q, k, v, _output, _lse = ctx.saved_tensors
        dq, dk, dv = _recompute_gradients(
            q,
            k,
            v,
            grad_output,
            ctx.is_causal,
            (ctx.needs_input_grad[0], ctx.needs_input_grad[1], ctx.needs_input_grad[2]),
        )
        return dq, dk, dv, None
