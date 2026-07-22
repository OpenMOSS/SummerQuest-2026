from __future__ import annotations

import math

import torch
from einops import rearrange
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, mean=0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.trunc_normal_(self.weight, mean=0, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Two-layer SiLU FFN used for the parameter-matched ablation."""

    def __init__(self, d_model: int, d_ff: int) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        normalized = x.float() * torch.rsqrt(x.float().square().mean(dim=-1, keepdim=True) + self.eps)
        return (normalized * self.weight.float()).to(dtype)


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int) -> None:
        super().__init__()
        if d_k % 2:
            raise ValueError("RoPE dimension must be even")
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        inv_freq = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x1, x2 = x[..., 0::2], x[..., 1::2]
        return rearrange(
            torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1), "... half pair -> ... (half pair)"
        )


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, dim=-1) @ v


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None) -> None:
        super().__init__()
        if d_model % num_heads:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        q, k, v = [
            rearrange(proj(x), "... seq (head d) -> ... head seq d", head=self.num_heads)
            for proj in (self.q_proj, self.k_proj, self.v_proj)
        ]
        seq_len = x.shape[-2]
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        if self.rope is not None:
            positions = rearrange(token_positions, "... seq -> ... 1 seq")
            q, k = self.rope(q, positions), self.rope(k, positions)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(q, k, v, mask)
        return self.output_proj(rearrange(attended, "... head seq d -> ... seq (head d)"))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        position_encoding: str = "rope",
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position must be 'pre' or 'post'")
        if position_encoding not in {"rope", "none"}:
            raise ValueError("position_encoding must be 'rope' or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_position = norm_position
        rope = (
            RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len) if position_encoding == "rope" else None
        )
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope)
        self.ffn = SwiGLU(d_model, d_ff) if ffn_type == "swiglu" else SiLUFeedForward(d_model, d_ff)
        self.ln1 = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.ln2 = RMSNorm(d_model) if use_rmsnorm else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        x = self.ln1(x + self.attn(x))
        return self.ln2(x + self.ffn(x))


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        position_encoding: str = "rope",
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rmsnorm,
                    norm_position,
                    position_encoding,
                    ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, indices: Tensor) -> Tensor:
        if indices.shape[-1] > self.context_length:
            raise ValueError("sequence length exceeds context length")
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
