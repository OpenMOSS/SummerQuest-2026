from __future__ import annotations

import math

import torch
from einops import einsum, rearrange
from torch import nn


class Linear(nn.Module):
    """Bias-free linear layer using row-major weights of shape (out, in)."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return einsum(x, self.weight, "... d_in, d_out d_in -> ... d_out")


class Embedding(nn.Module):
    """Token embedding table indexed by integer token ids."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square normalization over the final dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        in_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + self.eps)
        return (x_float / rms * self.weight.to(torch.float32)).to(in_dtype)


def silu(x: torch.Tensor) -> torch.Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise feed-forward network: W2(SiLU(W1 x) * W3 x)."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Two-layer SiLU feed-forward network for ablation runs."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class IdentityNorm(nn.Module):
    """Drop-in replacement used for no-RMSNorm ablations."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class RotaryPositionalEmbedding(nn.Module):
    """Rotary positional embedding applied pairwise to the final dimension."""

    def __init__(self, theta: float, d_k: int, max_seq_len: int, device=None):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even embedding dimension.")
        half_dim = d_k // 2
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        dims = torch.arange(half_dim, device=device, dtype=torch.float32)
        inv_freq = theta ** (-2 * dims / d_k)
        angles = positions[:, None] * inv_freq[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor) -> torch.Tensor:
        positions = token_positions.to(device=x.device, dtype=torch.long)
        while positions.ndim < x.ndim - 1:
            positions = positions.unsqueeze(-2)

        cos = self.cos[positions].to(dtype=x.dtype)
        sin = self.sin[positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        out = torch.empty_like(x)
        out[..., 0::2] = x_even * cos - x_odd * sin
        out[..., 1::2] = x_even * sin + x_odd * cos
        return out


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = q.shape[-1]
    scores = einsum(q, k, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask.to(device=scores.device, dtype=torch.bool), torch.finfo(scores.dtype).min)
    attn = softmax(scores, dim=-1)
    return einsum(attn, v, "... query key, ... key d_v -> ... query d_v")


class MultiheadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE on Q and K."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... seq_len (num_heads d_head) -> ... num_heads seq_len d_head", num_heads=self.num_heads)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... num_heads seq_len d_head -> ... seq_len (num_heads d_head)")

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        seq_len = x.shape[-2]
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool))
        attn_out = scaled_dot_product_attention(q, k, v, causal_mask)
        return self.output_proj(self._merge_heads(attn_out))


class TransformerBlock(nn.Module):
    """Transformer block with configurable norm and FFN variants."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        use_rmsnorm: bool = True,
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.post_norm = post_norm
        attn_theta = theta if use_rope else None
        attn_max_seq_len = max_seq_len if use_rope else None
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        self.attn = MultiheadSelfAttention(d_model, num_heads, attn_max_seq_len, attn_theta, device=device, dtype=dtype)
        self.ln1 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'.")
        self.ln2 = norm_cls(d_model, device=device, dtype=dtype) if use_rmsnorm else norm_cls()

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        if self.post_norm:
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model returning next-token logits."""

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
        post_norm: bool = False,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(d_model, num_heads, d_ff, context_length, rope_theta, device=device, dtype=dtype)
                if use_rmsnorm and not post_norm and use_rope and ffn_type == "swiglu"
                else TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    post_norm=post_norm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if use_rmsnorm else IdentityNorm()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("Input sequence length exceeds the configured context length.")
        token_positions = torch.arange(seq_len, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))
