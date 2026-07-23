"""Transformer LM building blocks. Implemented without torch.nn / torch.nn.functional cores."""

from __future__ import annotations

import math
import torch
from torch import Tensor, nn


# --- primitives ---


def softmax(x: Tensor, dim: int) -> Tensor:
    m = x.amax(dim=dim, keepdim=True)
    e = (x - m).exp()
    return e / e.sum(dim=dim, keepdim=True)


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean CE. logits (..., V), targets (...) with class indices."""
    logits = logits.view(-1, logits.size(-1))
    targets = targets.view(-1)
    m = logits.amax(dim=-1, keepdim=True)
    lse = m.squeeze(-1) + (logits - m).exp().sum(-1).log()
    tgt = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (lse - tgt).mean()


# --- modules ---


class Linear(nn.Module):
    """y = x @ W^T. No bias."""

    def __init__(self, d_in: int, d_out: int, device=None, dtype=None):
        super().__init__()
        std = math.sqrt(2.0 / (d_in + d_out))
        w = torch.empty(d_out, d_in, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, std=std, a=-3 * std, b=3 * std)
        self.weight = nn.Parameter(w)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        w = torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        nn.init.trunc_normal_(w, std=1.0, a=-3, b=3)
        self.weight = nn.Parameter(w)

    def forward(self, ids: Tensor) -> Tensor:
        return self.weight[ids]


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        orig = x.dtype
        x32 = x.to(torch.float32)
        rms = x32.pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x32 * rms).to(orig) * self.weight


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class RoPE(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0
        half = d_k // 2
        inv_freq = 1.0 / (theta ** (torch.arange(0, half, dtype=torch.float32, device=device) / half))
        pos = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        freqs = pos.unsqueeze(-1) * inv_freq.unsqueeze(0)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x: Tensor, positions: Tensor) -> Tensor:
        cos = self.cos[positions]
        sin = self.sin[positions]
        x1, x2 = x[..., 0::2], x[..., 1::2]
        while cos.dim() < x1.dim():  # unsqueeze so it broadcasts across heads
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.stack([rot1, rot2], dim=-1).flatten(-2)
        return out.to(x.dtype)


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    d_k = q.size(-1)
    scores = q @ k.transpose(-2, -1) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        *lead, seq, _ = x.shape
        q = self.q_proj(x).reshape(*lead, seq, self.num_heads, self.d_head).transpose(-2, -3)
        k = self.k_proj(x).reshape(*lead, seq, self.num_heads, self.d_head).transpose(-2, -3)
        v = self.v_proj(x).reshape(*lead, seq, self.num_heads, self.d_head).transpose(-2, -3)
        if self.rope is not None:
            if positions is None:
                positions = torch.arange(seq, device=x.device).expand(*lead, seq)
            q = self.rope(q, positions)
            k = self.rope(k, positions)
        causal = torch.tril(torch.ones(seq, seq, dtype=torch.bool, device=x.device))
        y = scaled_dot_product_attention(q, k, v, mask=causal)
        y = y.transpose(-2, -3).reshape(*lead, seq, self.d_model)
        return self.output_proj(y)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        rope: RoPE | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if rope is None:
            rope = RoPE(theta, d_model // num_heads, max_seq_len, device=device)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor, positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), positions)
        x = x + self.ffn(self.ln2(x))
        return x


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
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = RoPE(rope_theta, d_model // num_heads, context_length, device=device)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, rope=rope, device=device, dtype=dtype)
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, ids: Tensor) -> Tensor:
        x = self.token_embeddings(ids)
        positions = torch.arange(ids.size(-1), device=ids.device).expand_as(ids)
        for blk in self.layers:
            x = blk(x, positions)
        x = self.ln_final(x)
        return self.lm_head(x)
