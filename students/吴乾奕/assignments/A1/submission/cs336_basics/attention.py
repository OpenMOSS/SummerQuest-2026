"""Rotary embeddings and causal multi-head self-attention from tensor ops."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .nn import Linear


def softmax(inputs: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim``.

    An all-``-inf`` slice can occur for an entirely masked attention row.  Such a
    row is defined to produce zeros rather than NaNs.
    """

    output_dtype = inputs.dtype
    working = inputs.float() if output_dtype in {torch.float16, torch.bfloat16} else inputs
    maxima = working.amax(dim=dim, keepdim=True)
    shifted = torch.where(torch.isfinite(maxima), working - maxima, working)
    exponentials = torch.exp(shifted)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    return (exponentials / safe_denominator).to(output_dtype)


stable_softmax = softmax


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention for arbitrary leading dimensions.

    ``mask`` follows the assignment convention: ``True`` entries are visible and
    ``False`` entries are hidden.
    """

    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError(f"query/key dimensions must match, got {queries.shape[-1]} and {keys.shape[-1]}")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError(f"key/value sequence lengths must match, got {keys.shape[-2]} and {values.shape[-2]}")

    scale = math.sqrt(queries.shape[-1])
    scores = queries @ keys.transpose(-1, -2) / scale
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError(f"attention mask must have bool dtype, got {mask.dtype}")
        scores = scores.masked_fill(~mask, -torch.inf)
    attention_weights = softmax(scores, dim=-1)
    return attention_weights @ values


class RotaryPositionalEmbedding(nn.Module):
    """Rotary position embedding for adjacent pairs in the final dimension."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError(f"theta must be positive, got {theta}")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError(f"RoPE dimension must be a positive even number, got {d_k}")
        if max_seq_len <= 0:
            raise ValueError(f"max_seq_len must be positive, got {max_seq_len}")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dimension_pairs = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimension_pairs / d_k)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # These are deterministic caches, not model state.  Marking them
        # non-persistent keeps checkpoint keys compatible with the reference.
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if inputs.shape[-1] != self.d_k:
            raise ValueError(f"expected RoPE input dimension {self.d_k}, got {inputs.shape[-1]}")

        sequence_length = inputs.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=inputs.device)
        else:
            token_positions = token_positions.to(device=inputs.device)
        if token_positions.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise TypeError(f"token_positions must contain integers, got {token_positions.dtype}")
        if token_positions.shape[-1] != sequence_length:
            raise ValueError(
                "token_positions and inputs must have the same sequence length, "
                f"got {token_positions.shape[-1]} and {sequence_length}"
            )

        cosine = self.cos_cache[token_positions].to(dtype=inputs.dtype)
        sine = self.sin_cache[token_positions].to(dtype=inputs.dtype)
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]

        # Q/K often contain a head dimension that token_positions does not.
        # Insert singleton dimensions immediately before the sequence dimension
        # until standard broadcasting lines the tensors up.
        while cosine.ndim < even.ndim:
            cosine = cosine.unsqueeze(-3)
            sine = sine.unsqueeze(-3)

        rotated_even = even * cosine - odd * sine
        rotated_odd = even * sine + odd * cosine
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


class MultiHeadSelfAttention(nn.Module):
    """Batched causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError(f"d_model and num_heads must be positive, got {d_model} and {num_heads}")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        if rope is not None and rope.d_k != self.d_head:
            raise ValueError(f"RoPE dimension must equal head dimension {self.d_head}, got {rope.d_k}")
        self.rope = rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, inputs: Tensor) -> Tensor:
        sequence_length = inputs.shape[-2]
        split_shape = (*inputs.shape[:-2], sequence_length, self.num_heads, self.d_head)
        return inputs.reshape(split_shape).transpose(-3, -2)

    def _merge_heads(self, inputs: Tensor) -> Tensor:
        sequence_length = inputs.shape[-2]
        merged = inputs.transpose(-3, -2).contiguous()
        return merged.reshape(*merged.shape[:-3], sequence_length, self.d_model)

    def forward(
        self,
        inputs: Tensor,
        token_positions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"expected final input dimension {self.d_model}, got {inputs.shape[-1]}")

        queries = self._split_heads(self.q_proj(inputs))
        keys = self._split_heads(self.k_proj(inputs))
        values = self._split_heads(self.v_proj(inputs))

        if self.rope is not None:
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        sequence_length = inputs.shape[-2]
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=inputs.device,
        ).tril()
        if mask is not None:
            if mask.dtype != torch.bool:
                raise TypeError(f"attention mask must have bool dtype, got {mask.dtype}")
            # A per-batch (..., T, T) mask needs a singleton head axis.
            if mask.ndim == queries.ndim - 1:
                mask = mask.unsqueeze(-3)
            causal_mask = causal_mask & mask

        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._merge_heads(attended))


__all__ = [
    "MultiHeadSelfAttention",
    "RotaryPositionalEmbedding",
    "scaled_dot_product_attention",
    "softmax",
    "stable_softmax",
]
