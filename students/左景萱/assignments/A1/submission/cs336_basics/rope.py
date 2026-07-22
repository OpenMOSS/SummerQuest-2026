"""Rotary positional embeddings."""

from __future__ import annotations

import torch
from torch import Tensor, nn


class RotaryPositionalEmbedding(nn.Module):
    """Apply RoPE to adjacent pairs in the final feature dimension."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, *, device=None) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = float(theta)
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inverse_frequencies)
        # These values are deterministic caches rather than learned/checkpointed state.
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, inputs: Tensor, token_positions: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_k:
            raise ValueError(f"expected final dimension {self.d_k}, got {inputs.shape[-1]}")
        if token_positions.shape[-1] != inputs.shape[-2]:
            raise ValueError("token_positions and inputs must have the same sequence length")
        if token_positions.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_positions must be an integer tensor")
        if not torch.compiler.is_compiling() and token_positions.numel() and (
            bool((token_positions < 0).any()) or bool((token_positions >= self.max_seq_len).any())
        ):
            raise ValueError(f"token positions must be in [0, {self.max_seq_len})")

        cos_cache = self.get_buffer("cos")
        sin_cache = self.get_buffer("sin")
        positions = token_positions.to(device=cos_cache.device, dtype=torch.long)
        cos = cos_cache[positions]
        sin = sin_cache[positions]
        # token_positions may omit head and/or batch dimensions. Insert singleton
        # dimensions immediately before sequence until broadcasting is possible.
        while cos.ndim < inputs.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        cos = cos.to(device=inputs.device, dtype=inputs.dtype)
        sin = sin.to(device=inputs.device, dtype=inputs.dtype)

        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


# A short alias is convenient in configs and external training scripts.
RoPE = RotaryPositionalEmbedding

__all__ = ["RoPE", "RotaryPositionalEmbedding"]
