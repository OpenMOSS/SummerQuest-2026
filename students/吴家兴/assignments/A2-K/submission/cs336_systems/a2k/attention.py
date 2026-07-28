"""Explicit and tiled PyTorch attention used by the A2-K experiments."""

from __future__ import annotations

import math

import torch


def validate_attention_inputs(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
) -> None:
    """Validate the common ``[..., sequence, head_dim]`` attention contract."""

    if q.ndim < 3 or k.ndim != q.ndim or v.ndim != q.ndim:
        raise ValueError("Q, K, and V must have matching rank >= 3")
    if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
        raise ValueError("Q, K, and V must have matching leading dimensions")
    if q.shape[-1] != k.shape[-1] or q.shape[-1] != v.shape[-1]:
        raise ValueError("Q, K, and V must use the same head dimension")
    if k.shape[-2] != v.shape[-2]:
        raise ValueError("K and V must use the same sequence length")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q, K, and V must be on the same device")
    if q.dtype != k.dtype or q.dtype != v.dtype:
        raise ValueError("Q, K, and V must use the same dtype")


def causal_mask(
    n_queries: int,
    n_keys: int,
    device: torch.device,
) -> torch.Tensor:
    """Return the explicit self-attention mask used by the unfused baseline."""

    query_positions = torch.arange(n_queries, device=device)
    key_positions = torch.arange(n_keys, device=device)
    return query_positions[:, None] >= key_positions[None, :]


def explicit_attention_with_lse(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute explicit ``QK^T -> mask -> softmax -> PV`` attention.

    This intentionally materializes the score and probability matrices.  It
    does not call scaled-dot-product attention or any fused implementation.
    """

    validate_attention_inputs(q, k, v)
    scale = 1.0 / math.sqrt(q.shape[-1])
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    if is_causal:
        scores = scores.masked_fill(
            ~causal_mask(q.shape[-2], k.shape[-2], q.device),
            float("-inf"),
        )
    probabilities = torch.softmax(scores, dim=-1)
    output = torch.matmul(probabilities, v)
    lse = torch.logsumexp(scores.float(), dim=-1)
    return output, lse


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Return only the output of the explicit, unfused attention baseline."""

    return explicit_attention_with_lse(q, k, v, is_causal)[0]


def tiled_attention_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
    *,
    query_tile_size: int = 64,
    key_tile_size: int = 128,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Pure-PyTorch online-softmax attention without a full score matrix."""

    validate_attention_inputs(q, k, v)
    if query_tile_size <= 0 or key_tile_size <= 0:
        raise ValueError("tile sizes must be positive")

    leading_shape = q.shape[:-2]
    n_queries, head_dim = q.shape[-2:]
    n_keys = k.shape[-2]
    flat_batch = math.prod(leading_shape)
    q_flat = q.reshape(flat_batch, n_queries, head_dim)
    k_flat = k.reshape(flat_batch, n_keys, head_dim)
    v_flat = v.reshape(flat_batch, n_keys, head_dim)

    output = torch.empty_like(q_flat)
    lse = torch.empty(
        (flat_batch, n_queries),
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(head_dim)

    for query_start in range(0, n_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, n_queries)
        rows = query_end - query_start
        query_tile = q_flat[:, query_start:query_end].float()
        running_max = torch.full(
            (flat_batch, rows),
            -float("inf"),
            device=q.device,
            dtype=torch.float32,
        )
        running_sum = torch.zeros_like(running_max)
        accumulator = torch.zeros(
            (flat_batch, rows, head_dim),
            device=q.device,
            dtype=torch.float32,
        )

        query_positions = torch.arange(
            query_start,
            query_end,
            device=q.device,
        )
        for key_start in range(0, n_keys, key_tile_size):
            key_end = min(key_start + key_tile_size, n_keys)
            key_tile = k_flat[:, key_start:key_end].float()
            value_tile = v_flat[:, key_start:key_end].float()
            scores = torch.matmul(
                query_tile,
                key_tile.transpose(-2, -1),
            ) * scale

            if is_causal:
                key_positions = torch.arange(
                    key_start,
                    key_end,
                    device=q.device,
                )
                valid = query_positions[:, None] >= key_positions[None, :]
                scores = scores.masked_fill(~valid, -float("inf"))

            tile_max = scores.amax(dim=-1)
            new_max = torch.maximum(running_max, tile_max)
            correction = torch.exp(running_max - new_max)
            unnormalized = torch.exp(scores - new_max.unsqueeze(-1))
            if is_causal:
                unnormalized = torch.where(
                    valid,
                    unnormalized,
                    torch.zeros_like(unnormalized),
                )
            running_sum = (
                correction * running_sum + unnormalized.sum(dim=-1)
            )
            accumulator = (
                correction.unsqueeze(-1) * accumulator
                + torch.matmul(unnormalized, value_tile)
            )
            running_max = new_max

        normalized = accumulator / running_sum.clamp_min(1e-20).unsqueeze(-1)
        output[:, query_start:query_end] = normalized.to(q.dtype)
        lse[:, query_start:query_end] = (
            running_max + torch.log(running_sum.clamp_min(1e-20))
        )

    return (
        output.reshape(*leading_shape, n_queries, head_dim),
        lse.reshape(*leading_shape, n_queries),
    )


def tiled_attention_backward(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    grad_output: torch.Tensor,
    is_causal: bool,
    *,
    query_tile_size: int = 256,
    key_tile_size: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Recompute probabilities tile-wise and return ``dQ``, ``dK``, ``dV``."""

    validate_attention_inputs(q, k, v)
    leading_shape = q.shape[:-2]
    n_queries, head_dim = q.shape[-2:]
    n_keys = k.shape[-2]
    flat_batch = math.prod(leading_shape)
    q_flat = q.reshape(flat_batch, n_queries, head_dim)
    k_flat = k.reshape(flat_batch, n_keys, head_dim)
    v_flat = v.reshape(flat_batch, n_keys, head_dim)
    output_flat = output.reshape(flat_batch, n_queries, head_dim)
    grad_output_flat = grad_output.reshape(flat_batch, n_queries, head_dim)
    lse_flat = lse.reshape(flat_batch, n_queries)

    grad_q = torch.zeros_like(q_flat, dtype=torch.float32)
    grad_k = torch.zeros_like(k_flat, dtype=torch.float32)
    grad_v = torch.zeros_like(v_flat, dtype=torch.float32)
    scale = 1.0 / math.sqrt(head_dim)

    for query_start in range(0, n_queries, query_tile_size):
        query_end = min(query_start + query_tile_size, n_queries)
        query_tile = q_flat[:, query_start:query_end].float()
        output_tile = output_flat[:, query_start:query_end].float()
        grad_output_tile = grad_output_flat[:, query_start:query_end].float()
        lse_tile = lse_flat[:, query_start:query_end].float()
        delta = (output_tile * grad_output_tile).sum(dim=-1)
        grad_query_tile = torch.zeros_like(query_tile)
        query_positions = torch.arange(
            query_start,
            query_end,
            device=q.device,
        )

        for key_start in range(0, n_keys, key_tile_size):
            key_end = min(key_start + key_tile_size, n_keys)
            key_tile = k_flat[:, key_start:key_end].float()
            value_tile = v_flat[:, key_start:key_end].float()
            scores = torch.matmul(
                query_tile,
                key_tile.transpose(-2, -1),
            ) * scale

            if is_causal:
                key_positions = torch.arange(
                    key_start,
                    key_end,
                    device=q.device,
                )
                valid = query_positions[:, None] >= key_positions[None, :]
                scores = scores.masked_fill(~valid, -float("inf"))
            else:
                valid = None

            probabilities = torch.exp(scores - lse_tile.unsqueeze(-1))
            if valid is not None:
                probabilities = torch.where(
                    valid,
                    probabilities,
                    torch.zeros_like(probabilities),
                )
            grad_probabilities = torch.matmul(
                grad_output_tile,
                value_tile.transpose(-2, -1),
            )
            grad_scores = probabilities * (
                grad_probabilities - delta.unsqueeze(-1)
            )
            grad_query_tile += torch.matmul(grad_scores, key_tile) * scale
            grad_k[:, key_start:key_end] += (
                torch.matmul(grad_scores.transpose(-2, -1), query_tile)
                * scale
            )
            grad_v[:, key_start:key_end] += torch.matmul(
                probabilities.transpose(-2, -1),
                grad_output_tile,
            )

        grad_q[:, query_start:query_end] = grad_query_tile

    return (
        grad_q.reshape_as(q).to(q.dtype),
        grad_k.reshape_as(k).to(k.dtype),
        grad_v.reshape_as(v).to(v.dtype),
    )


class FlashAttentionPyTorch(torch.autograd.Function):
    """Pure-PyTorch tiled FlashAttention with recomputation backward."""

    @staticmethod
    def forward(
        ctx: torch.autograd.function.FunctionCtx,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        is_causal: bool = False,
    ) -> torch.Tensor:
        output, lse = tiled_attention_forward(
            q,
            k,
            v,
            bool(is_causal),
        )
        ctx.is_causal = bool(is_causal)
        ctx.save_for_backward(q, k, v, output, lse)
        return output

    @staticmethod
    def backward(
        ctx: torch.autograd.function.FunctionCtx,
        grad_output: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, None]:
        q, k, v, output, lse = ctx.saved_tensors
        grad_q, grad_k, grad_v = tiled_attention_backward(
            q,
            k,
            v,
            output,
            lse,
            grad_output,
            ctx.is_causal,
        )
        return grad_q, grad_k, grad_v, None
