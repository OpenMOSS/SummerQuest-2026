from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from .nn import Linear


def softmax(x: Tensor, dim: int) -> Tensor:
    input_dtype = x.dtype
    if input_dtype in {torch.float16, torch.bfloat16}:
        x = x.to(torch.float32)
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    result = exponentials / torch.sum(exponentials, dim=dim, keepdim=True)
    return result.to(input_dtype)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = torch.einsum("...qd,...kd->...qk", q, k) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kv->...qv", probabilities, v)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even query/key dimension")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.einsum("s,d->sd", positions, inverse_frequencies)
        self.cos: Tensor
        self.sin: Tensor
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if (
            not torch.compiler.is_compiling()
            and token_positions.numel()
            and int(token_positions.max()) >= self.max_seq_len
        ):
            raise ValueError("token position exceeds the configured maximum sequence length")
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]
        while cos.ndim < even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        output = torch.empty_like(x)
        output[..., 0::2] = even * cos - odd * sin
        output[..., 1::2] = even * sin + odd * cos
        return output


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
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
        self.max_seq_len = max_seq_len
        self.use_rope = use_rope
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device) if use_rope else None
        )
        self.causal_mask: Tensor
        self.register_buffer(
            "causal_mask",
            torch.tril(torch.ones(max_seq_len, max_seq_len, device=device, dtype=torch.bool)),
            persistent=False,
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        return x.reshape(*x.shape[:-2], sequence_length, self.num_heads, self.head_dim).transpose(-3, -2)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        if sequence_length > self.max_seq_len:
            raise ValueError("input sequence is longer than max_seq_len")
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal_mask = self.causal_mask[:sequence_length, :sequence_length]
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        attended = attended.transpose(-3, -2).contiguous().reshape(*x.shape[:-2], sequence_length, self.d_model)
        return self.output_proj(attended)
