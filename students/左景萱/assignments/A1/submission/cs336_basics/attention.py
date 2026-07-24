"""Scaled dot-product and causal multi-head self-attention."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .nn import Linear
from .rope import RotaryPositionalEmbedding


def softmax(inputs: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim``."""

    shifted = inputs - inputs.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(query: Tensor, key: Tensor, value: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute scaled dot-product attention for arbitrary leading dimensions.

    A boolean mask uses ``True`` for entries that may be attended to.
    """

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key dimensions must match")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value sequence lengths must match")

    scores = query @ key.transpose(-1, -2)
    scores = scores / math.sqrt(query.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must be boolean")
        scores = scores.masked_fill(~mask, -torch.inf)

    # Softmax in fp32 avoids overflow/underflow during mixed-precision training.
    probabilities = softmax(scores.float(), dim=-1).to(value.dtype)
    return probabilities @ value


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be evenly divisible by num_heads")
        if use_rope and max_seq_len is None:
            raise ValueError("max_seq_len is required when RoPE is enabled")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if use_rope and max_seq_len is not None
            else None
        )

    def _split_heads(self, tensor: Tensor) -> Tensor:
        shape = (*tensor.shape[:-1], self.num_heads, self.head_dim)
        # (..., sequence, heads, head_dim) -> (..., heads, sequence, head_dim)
        return tensor.reshape(shape).transpose(-3, -2)

    def _join_heads(self, tensor: Tensor) -> Tensor:
        # (..., heads, sequence, head_dim) -> (..., sequence, d_model)
        return tensor.transpose(-3, -2).contiguous().flatten(-2)

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {inputs.shape[-1]}")
        sequence_length = inputs.shape[-2]
        if self.max_seq_len is not None and sequence_length > self.max_seq_len:
            raise ValueError(f"sequence length {sequence_length} exceeds maximum {self.max_seq_len}")

        query = self._split_heads(self.q_proj(inputs))
        key = self._split_heads(self.k_proj(inputs))
        value = self._split_heads(self.v_proj(inputs))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=inputs.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            device=inputs.device,
            dtype=torch.bool,
        ).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        return self.output_proj(self._join_heads(attended))


__all__ = ["MultiHeadSelfAttention", "scaled_dot_product_attention", "softmax"]
