from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cs336_basics.modules import Linear


class RoPE(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE dimension must be even")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")
        inverse_frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError("input final dimension does not match RoPE dimension")
        if token_positions.numel() and int(token_positions.max()) >= self.max_seq_len:
            raise ValueError("token position exceeds the configured maximum sequence length")

        positions = token_positions.long()
        while positions.ndim < x.ndim - 1:
            positions = positions.unsqueeze(-2)
        cos = self.cos[positions].to(device=x.device, dtype=x.dtype)
        sin = self.sin[positions].to(device=x.device, dtype=x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    scores = queries @ keys.transpose(-1, -2) / math.sqrt(queries.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))

    row_max = scores.max(dim=-1, keepdim=True).values
    safe_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
    weights = torch.exp(scores - safe_max)
    if mask is not None:
        weights = torch.where(mask, weights, torch.zeros_like(weights))
    denominator = weights.sum(dim=-1, keepdim=True)
    weights = torch.where(denominator > 0, weights / denominator.clamp_min(torch.finfo(weights.dtype).tiny), weights)
    return weights @ values


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        use_rope: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RoPE(theta=theta, d_k=self.head_dim, max_seq_len=max_seq_len, device=device)
            if use_rope and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        leading = x.shape[:-2]
        x = x.reshape(*leading, sequence_length, self.num_heads, self.head_dim)
        return x.transpose(-3, -2)

    def _join_heads(self, x: Tensor) -> Tensor:
        x = x.transpose(-3, -2)
        return x.reshape(*x.shape[:-2], self.d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        causal_mask = torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device))
        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._join_heads(attended))
