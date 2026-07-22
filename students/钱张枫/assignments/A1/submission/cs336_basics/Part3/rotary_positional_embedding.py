from __future__ import annotations

import torch
from einops import rearrange
from torch import nn


class RotaryPositionalEmbedding(nn.Module):
    """Rotary Positional Embedding，对 query/key 的最后一维按相邻二元组旋转。"""

    theta: float
    d_k: int
    max_seq_len: int

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even d_k so dimensions can be rotated in pairs.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        # inv_freq[i] = theta^(-2i / d_k)，对应 RoPE 中每一对 hidden 维度的旋转频率。
        dim_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inv_freq = theta ** (-dim_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        # token_positions 用于按真实位置索引预计算表，支持形状为 (..., seq_len)。
        cos = self.cos[token_positions].to(dtype=x.dtype, device=x.device)
        sin = self.sin[token_positions].to(dtype=x.dtype, device=x.device)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        # 若 x 比 token_positions 多出 batch-like 维度，在 sequence 维之前补 1 以便广播。
        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        return rearrange(torch.stack((rotated_even, rotated_odd), dim=-1), "... half pair -> ... (half pair)")
