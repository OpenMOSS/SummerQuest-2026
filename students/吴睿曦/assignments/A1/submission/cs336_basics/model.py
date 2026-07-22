import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        nn.init.trunc_normal_(self.weight, std=math.sqrt(2 / (d_in + d_out)), a=-3, b=3)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.T


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        nn.init.trunc_normal_(self.weight, std=1, a=-3, b=3)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - x.max(dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / exp.sum(dim=dim, keepdim=True)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))

    def forward(self, x: Tensor) -> Tensor:
        dtype = x.dtype
        xf = x.float()
        result = xf * torch.rsqrt(xf.square().mean(dim=-1, keepdim=True) + self.eps)
        return (result * self.weight.float()).to(dtype)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, d_k, 2, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        angles = torch.outer(positions, inv_freq)
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        even, odd = x[..., 0::2], x[..., 1::2]
        return torch.stack((even * cos - odd * sin, even * sin + odd * cos), dim=-1).flatten(-2)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-2, -1) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, -torch.inf)
    return softmax(scores, -1) @ v


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope: RotaryPositionalEmbedding | None = None):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)
        self.rope = rope

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        *leading, seq_len, _ = x.shape
        def split(proj):
            return proj(x).view(*leading, seq_len, self.num_heads, self.head_dim).transpose(-3, -2)
        q, k, v = split(self.q_proj), split(self.k_proj), split(self.v_proj)
        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q, k = self.rope(q, token_positions), self.rope(k, token_positions)
        mask = torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device).tril()
        out = scaled_dot_product_attention(q, k, v, mask)
        out = out.transpose(-3, -2).contiguous().view(*leading, seq_len, -1)
        return self.output_proj(out)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, d_ff: int, max_seq_len: int, theta: float,
        norm_style: str = "pre", use_rope: bool = True, ffn_variant: str = "swiglu",
    ):
        super().__init__()
        if norm_style not in {"pre", "post", "none"}:
            raise ValueError("norm_style must be pre, post, or none")
        rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len) if use_rope else None
        self.attn = MultiheadSelfAttention(d_model, num_heads, rope)
        self.ln1 = RMSNorm(d_model) if norm_style != "none" else Identity()
        self.ffn = SwiGLU(d_model, d_ff) if ffn_variant == "swiglu" else SiLUFeedForward(d_model, d_ff)
        self.ln2 = RMSNorm(d_model) if norm_style != "none" else Identity()
        self.norm_style = norm_style

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_style == "post":
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    def __init__(
        self, vocab_size: int, context_length: int, d_model: int, num_layers: int,
        num_heads: int, d_ff: int, rope_theta: float, norm_style: str = "pre",
        use_rope: bool = True, ffn_variant: str = "swiglu",
    ):
        super().__init__()
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, norm_style, use_rope, ffn_variant)
            for _ in range(num_layers)
        ])
        self.ln_final = RMSNorm(d_model) if norm_style != "none" else Identity()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, indices: Tensor) -> Tensor:
        x = self.token_embeddings(indices)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
