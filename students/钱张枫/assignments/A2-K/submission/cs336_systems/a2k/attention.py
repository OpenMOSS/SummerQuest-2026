"""Explicit and tiled attention primitives used throughout A2-K.

No function in this module calls ``scaled_dot_product_attention`` or a
third-party fused attention implementation.  The full-matrix implementation is
an intentionally explicit eager baseline; the tiled implementation maintains
the online-softmax state required by FlashAttention.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class AttentionShape:
    """Validated dimensions for a batch-major attention invocation."""

    batch_size: int
    n_queries: int
    n_keys: int
    head_dim: int


def validate_attention_inputs(q: Tensor, k: Tensor, v: Tensor) -> AttentionShape:
    """Validate the ``[batch, sequence, head_dim]`` attention contract.

    The assignment interface deliberately uses one attention head per tensor.
    Keeping the boundary strict prevents silently benchmarking an unintended
    layout or mixed-precision combination.
    """

    tensors = {"q": q, "k": k, "v": v}
    for name, tensor in tensors.items():
        if tensor.ndim != 3:
            raise ValueError(f"{name} must have shape [batch, sequence, head_dim], got {tuple(tensor.shape)}")
        if not tensor.is_floating_point():
            raise TypeError(f"{name} must be floating point, got {tensor.dtype}")

    if q.device != k.device or q.device != v.device:
        raise ValueError("q, k, and v must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("q, k, and v must use the same dtype")
    if q.shape[0] != k.shape[0] or q.shape[0] != v.shape[0]:
        raise ValueError("q, k, and v must have the same batch size")
    if q.shape[2] != k.shape[2] or q.shape[2] != v.shape[2]:
        raise ValueError("q, k, and v must have the same head dimension")
    if k.shape[1] != v.shape[1]:
        raise ValueError("k and v must have the same sequence length")
    if q.shape[1] == 0 or k.shape[1] == 0:
        raise ValueError("attention does not support empty query or key sequences")

    return AttentionShape(
        batch_size=q.shape[0],
        n_queries=q.shape[1],
        n_keys=k.shape[1],
        head_dim=q.shape[2],
    )


def _accumulator_dtype(dtype: torch.dtype) -> torch.dtype:
    """Use FP32 accumulation for the assignment's FP16/BF16/FP32 paths."""

    if dtype in (torch.float16, torch.bfloat16, torch.float32):
        return torch.float32
    return dtype


def _causal_mask(
    query_start: int,
    query_end: int,
    key_start: int,
    key_end: int,
    *,
    device: torch.device,
) -> Tensor:
    """Return the assignment's causal mask: a query at ``i`` sees keys ``j <= i``."""

    query_positions = torch.arange(query_start, query_end, device=device)[:, None]
    key_positions = torch.arange(key_start, key_end, device=device)[None, :]
    return query_positions >= key_positions


def _materialized_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool,
    *,
    use_input_dtype_for_gemm: bool,
    return_lse: bool,
) -> tuple[Tensor, Tensor | None]:
    """Run materialized attention with either native or reference GEMMs."""

    shape = validate_attention_inputs(q, k, v)
    accumulation_dtype = _accumulator_dtype(q.dtype)
    if use_input_dtype_for_gemm:
        # Keep the benchmarked BF16 GEMM in BF16.  ``softmax(..., dtype=...)``
        # performs the numerically stable FP32 normalization without retaining
        # a second FP32 copy of the full score matrix.
        scores = torch.matmul(q, k.transpose(-1, -2))
        values = v
        probability_dtype = q.dtype
    else:
        # Correctness comparisons deliberately accumulate both GEMMs in FP32.
        # This is a numerical reference, not the timed BF16 baseline.
        q_work = q.to(accumulation_dtype)
        k_work = k.to(accumulation_dtype)
        values = v.to(accumulation_dtype)
        scores = torch.matmul(q_work, k_work.transpose(-1, -2))
        probability_dtype = accumulation_dtype

    # ``scores`` is a fresh matmul result, so scaling it in place avoids a
    # second full [batch, query, key] allocation. The operation remains part
    # of the autograd graph because the tensor is non-leaf.
    scores.mul_(1.0 / math.sqrt(shape.head_dim))
    if is_causal:
        valid = _causal_mask(0, shape.n_queries, 0, shape.n_keys, device=q.device)
        # The mask is consumed immediately by softmax/LSE; in-place masking
        # avoids retaining an additional score matrix at the peak.
        scores.masked_fill_(~valid.unsqueeze(0), float("-inf"))

    if use_input_dtype_for_gemm:
        probabilities = torch.softmax(scores, dim=-1, dtype=accumulation_dtype)
    else:
        probabilities = torch.softmax(scores, dim=-1)
    if return_lse:
        lse = torch.logsumexp(scores.to(accumulation_dtype), dim=-1)
    else:
        lse = None
    del scores
    output = torch.matmul(probabilities.to(probability_dtype), values).to(q.dtype)
    # The timed explicit baseline only consumes the output. Avoid building a
    # second autograd branch for LSE in that path; the reference helper below
    # still computes and returns LSE when correctness checks request it.
    return output, lse


def explicit_attention_with_lse(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> tuple[Tensor, Tensor]:
    """Compute an FP32-accumulating materialized attention reference and LSE."""

    output, lse = _materialized_attention(
        q,
        k,
        v,
        is_causal,
        use_input_dtype_for_gemm=False,
        return_lse=True,
    )
    assert lse is not None
    return output, lse


def explicit_attention(q: Tensor, k: Tensor, v: Tensor, is_causal: bool = False) -> Tensor:
    """Compute the timed explicit ``QK^T -> scale -> mask -> softmax -> PV`` baseline."""

    output, _ = _materialized_attention(
        q,
        k,
        v,
        is_causal,
        use_input_dtype_for_gemm=True,
        return_lse=False,
    )
    return output


def tiled_attention_forward(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    is_causal: bool = False,
    *,
    # 128x128 tiles amortize Python dispatch in the recomputation path while
    # keeping each FP32 score tile small (64 KiB per batch element).
    query_block_size: int = 128,
    key_block_size: int = 128,
) -> tuple[Tensor, Tensor]:
    """Compute tiled attention using numerically stable online softmax.

    For each query tile, ``row_max``, ``row_sum`` and the output accumulator
    are updated while key/value tiles stream through the calculation.  This is
    a correctness-oriented PyTorch reference, not a performance baseline.
    """

    if query_block_size <= 0 or key_block_size <= 0:
        raise ValueError("query_block_size and key_block_size must be positive")

    shape = validate_attention_inputs(q, k, v)
    accumulation_dtype = _accumulator_dtype(q.dtype)
    q_work = q.to(accumulation_dtype)
    k_work = k.to(accumulation_dtype)
    v_work = v.to(accumulation_dtype)
    scale = 1.0 / math.sqrt(shape.head_dim)

    output_chunks: list[Tensor] = []
    lse_chunks: list[Tensor] = []
    for query_start in range(0, shape.n_queries, query_block_size):
        query_end = min(query_start + query_block_size, shape.n_queries)
        q_tile = q_work[:, query_start:query_end, :]
        query_count = query_end - query_start
        row_max = torch.full(
            (shape.batch_size, query_count), float("-inf"), dtype=accumulation_dtype, device=q.device
        )
        row_sum = torch.zeros((shape.batch_size, query_count), dtype=accumulation_dtype, device=q.device)
        accumulator = torch.zeros(
            (shape.batch_size, query_count, shape.head_dim), dtype=accumulation_dtype, device=q.device
        )

        for key_start in range(0, shape.n_keys, key_block_size):
            key_end = min(key_start + key_block_size, shape.n_keys)
            k_tile = k_work[:, key_start:key_end, :]
            v_tile = v_work[:, key_start:key_end, :]
            scores = torch.matmul(q_tile, k_tile.transpose(-1, -2)) * scale
            if is_causal:
                valid = _causal_mask(query_start, query_end, key_start, key_end, device=q.device)
                scores = scores.masked_fill(~valid.unsqueeze(0), float("-inf"))

            tile_max = scores.max(dim=-1).values
            next_row_max = torch.maximum(row_max, tile_max)
            rescale_previous = torch.where(
                torch.isfinite(row_max), torch.exp(row_max - next_row_max), torch.zeros_like(row_max)
            )
            tile_probabilities = torch.exp(scores - next_row_max.unsqueeze(-1))
            row_sum = row_sum * rescale_previous + tile_probabilities.sum(dim=-1)
            accumulator = accumulator * rescale_previous.unsqueeze(-1) + torch.matmul(tile_probabilities, v_tile)
            row_max = next_row_max

        output_chunks.append(accumulator / row_sum.unsqueeze(-1))
        lse_chunks.append(row_max + torch.log(row_sum))

    output = torch.cat(output_chunks, dim=1).to(q.dtype)
    lse = torch.cat(lse_chunks, dim=1)
    return output, lse
