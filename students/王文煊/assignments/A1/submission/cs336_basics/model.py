"""Transformer language model components, implemented from scratch.

Only ``torch.nn.Parameter``, module containers (``Module``, ``ModuleList``) and
plain tensor ops are used -- no ``nn.Linear``/``nn.Embedding``/``F.*`` layers.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def silu(x: Tensor) -> Tensor:
    """SiLU / swish activation: x * sigmoid(x)."""
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim``."""
    x_max = torch.max(x, dim=dim, keepdim=True).values
    x = x - x_max
    x_exp = torch.exp(x)
    return x_exp / torch.sum(x_exp, dim=dim, keepdim=True)


class Linear(nn.Module):
    """y = x W^T, weight stored as (d_out, d_in), no bias."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """Learned lookup table of shape (vocab_size, d_model)."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square layer norm (no mean subtraction)."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        in_dtype = x.dtype
        x = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        out = (x / rms) * self.weight
        return out.to(in_dtype)


class Identity(nn.Module):
    """No-op stand-in used by the "remove RMSNorm" ablation."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class SwiGLU(nn.Module):
    """FFN(x) = W2 (SiLU(W1 x) * (W3 x))."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """Plain 2-layer SiLU FFN: W2 (SiLU(W1 x)). Used for the SwiGLU ablation."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE applied to consecutive dimension pairs (2k, 2k+1)."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        assert d_k % 2 == 0, "RoPE requires an even head dimension"
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len
        inv_freq = 1.0 / (theta ** (torch.arange(0, d_k, 2, dtype=torch.float32, device=device) / d_k))
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inv_freq)  # (max_seq_len, d_k/2)
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        # x: (..., seq_len, d_k)
        cos = self.cos_cached[token_positions]  # (..., seq_len, d_k/2)
        sin = self.sin_cached[token_positions]
        x1 = x[..., 0::2]
        x2 = x[..., 1::2]
        rot1 = x1 * cos - x2 * sin
        rot2 = x1 * sin + x2 * cos
        out = torch.empty_like(x)
        out[..., 0::2] = rot1
        out[..., 1::2] = rot2
        return out


def scaled_dot_product_attention(
    q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None
) -> Tensor:
    """Attention(Q,K,V) = softmax(QK^T / sqrt(d_k) + mask) V.

    ``mask`` is boolean: True = attend, False = forbidden (set to -inf).
    """
    d_k = q.shape[-1]
    scores = q @ k.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    attn = softmax(scores, dim=-1)
    return attn @ v


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        if use_rope:
            assert theta is not None and max_seq_len is not None
            self.rope = RotaryPositionalEmbedding(theta, self.d_k, max_seq_len, device=device)
        else:
            self.rope = None

    def _split_heads(self, x: Tensor) -> Tensor:
        # (..., seq, d_model) -> (..., heads, seq, d_k)
        *lead, seq, _ = x.shape
        x = x.view(*lead, seq, self.num_heads, self.d_k)
        return x.transpose(-2, -3)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *lead, seq_len, _ = x.shape
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
        out = scaled_dot_product_attention(q, k, v, mask=causal)

        out = out.transpose(-2, -3).contiguous().view(*lead, seq_len, self.d_model)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm (default) Transformer block with configurable ablations."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_rope: bool = True,
        norm: str = "rmsnorm",  # "rmsnorm" | "none"
        norm_position: str = "pre",  # "pre" | "post"
        ffn: str = "swiglu",  # "swiglu" | "silu"
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.norm_position = norm_position
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope=use_rope, device=device, dtype=dtype
        )
        if ffn == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn == "silu":
            self.ffn = SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown ffn type {ffn}")

        if norm == "rmsnorm":
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        elif norm == "none":
            self.ln1 = Identity()
            self.ln2 = Identity()
        else:
            raise ValueError(f"unknown norm type {norm}")

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_position == "pre":
            x = x + self.attn(self.ln1(x), token_positions)
            x = x + self.ffn(self.ln2(x))
        else:  # post-norm
            x = self.ln1(x + self.attn(x, token_positions))
            x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model returning logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        use_rope: bool = True,
        norm: str = "rmsnorm",
        norm_position: str = "pre",
        ffn: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rope=use_rope,
                    norm=norm,
                    norm_position=norm_position,
                    ffn=ffn,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        if norm == "rmsnorm":
            self.ln_final = RMSNorm(d_model, device=device, dtype=dtype)
        else:
            self.ln_final = Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.shape[-1]
        positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        x = self.ln_final(x)
        return self.lm_head(x)

    def num_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters())
