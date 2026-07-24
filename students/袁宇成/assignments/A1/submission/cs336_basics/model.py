from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        with torch.no_grad():
            std = math.sqrt(2.0 / (d_in + d_out))
            nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        with torch.no_grad():
            nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    numerator = shifted.exp()
    return numerator / numerator.sum(dim=dim, keepdim=True)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        normalized = x_float * torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(input_dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None, use_silu: bool = True):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.use_silu = use_silu

    def forward(self, x: Tensor) -> Tensor:
        gate = self.w1(x)
        if self.use_silu:
            gate = silu(gate)
        return self.w2(gate * self.w3(x))


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2:
            raise ValueError("RoPE requires an even head dimension")
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        angles = positions[:, None] * frequencies[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions]
        sin = self.sin[token_positions]
        paired_shape = x.shape[:-1] + (x.shape[-1] // 2, 2)
        pairs = x.float().reshape(paired_shape)
        first, second = pairs.unbind(dim=-1)
        while cos.ndim < first.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        rotated = torch.stack((first * cos - second * sin, first * sin + second * cos), dim=-1)
        return rotated.flatten(-2).to(x.dtype)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device) if theta else None

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        seq_len = x.shape[-2]
        batch_shape = x.shape[:-2]

        def split_heads(projected: Tensor) -> Tensor:
            return projected.reshape(*batch_shape, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)

        q = split_heads(self.q_proj(x))
        k = split_heads(self.k_proj(x))
        v = split_heads(self.v_proj(x))
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)
        causal = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(q, k, v, causal)
        attended = attended.transpose(-3, -2).contiguous().reshape(*batch_shape, seq_len, -1)
        return self.output_proj(attended)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device=None,
        dtype=None,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        use_silu: bool = True,
        ffn_variant: str = "swiglu",
        silu_d_ff: int | None = None,
    ):
        super().__init__()
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta if use_rope else None, device=device, dtype=dtype
        )
        if ffn_variant == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype, use_silu=use_silu)
        elif ffn_variant == "silu":
            self.ffn = SiLUFeedForward(d_model, silu_d_ff or round(1.5 * d_ff), device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown FFN variant: {ffn_variant}")
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.use_rmsnorm = use_rmsnorm
        self.post_norm = post_norm

    def _norm(self, norm: RMSNorm, x: Tensor) -> Tensor:
        return norm(x) if self.use_rmsnorm else x

    def forward(self, x: Tensor) -> Tensor:
        if self.post_norm:
            x = self._norm(self.ln1, x + self.attn(x))
            return self._norm(self.ln2, x + self.ffn(x))
        x = x + self.attn(self._norm(self.ln1, x))
        return x + self.ffn(self._norm(self.ln2, x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10000.0,
        device=None,
        dtype=None,
        **ablations,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                    **ablations,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("input sequence exceeds context length")
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
