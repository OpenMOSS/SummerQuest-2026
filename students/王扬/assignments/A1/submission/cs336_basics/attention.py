from __future__ import annotations

import torch
from torch import Tensor, nn

from cs336_basics.nn import Linear


def scaled_dot_product_attention(
    q: Tensor,
    k: Tensor,
    v: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    d_k = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-1, -2)) / torch.sqrt(torch.tensor(d_k, device=q.device, dtype=q.dtype))
    if mask is not None:
        scores = scores.masked_fill(~mask.to(device=q.device), torch.finfo(scores.dtype).min)
    attn_weights = torch.softmax(scores, dim=-1)
    return torch.matmul(attn_weights, v)


class RoPE(nn.Module):
    def __init__(self, d_k: int, theta: float, max_seq_len: int) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even head dimension.")

        self.d_k = d_k
        self.theta = theta
        self.max_seq_len = max_seq_len

        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32) / d_k))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        positions = token_positions.to(device=x.device)
        while positions.ndim < x.ndim - 1:
            positions = positions.unsqueeze(-2)
        angles = positions.unsqueeze(-1).to(dtype=x.dtype) * self.inv_freq.to(device=x.device, dtype=x.dtype)
        cos = torch.cos(angles)
        sin = torch.sin(angles)

        x_even = x[..., ::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., ::2] = rotated_even
        out[..., 1::2] = rotated_odd
        return out


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        use_rope: bool = False,
        rope_theta: float = 10000.0,
        max_seq_len: int = 0,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope
        self.rope_theta = rope_theta
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

        self.rope = (
            RoPE(d_k=self.d_k, theta=rope_theta, max_seq_len=max_seq_len)
            if use_rope
            else None
        )

    def _causal_mask(self, seq_len: int, device: torch.device) -> Tensor:
        return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))

    def _split_heads(self, x: Tensor) -> Tensor:
        *leading, seq_len, _ = x.shape
        return x.view(*leading, seq_len, self.num_heads, self.d_k).transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        *leading, num_heads, seq_len, d_k = x.shape
        if num_heads != self.num_heads or d_k != self.d_k:
            raise ValueError("Unexpected head shape while merging attention heads.")
        return x.transpose(-3, -2).contiguous().view(*leading, seq_len, self.d_model)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        *leading, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.use_rope:
            if self.rope is None:
                raise ValueError("RoPE module is not initialized.")
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
                if leading:
                    token_positions = token_positions.view(*([1] * len(leading)), seq_len).expand(*leading, seq_len)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attn_mask = mask
        if attn_mask is None:
            attn_mask = self._causal_mask(seq_len, x.device)

        attn_output = scaled_dot_product_attention(q, k, v, mask=attn_mask)
        merged = self._merge_heads(attn_output)
        return self.output_proj(merged)
