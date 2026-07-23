from __future__ import annotations

import math

import torch
from torch import nn


def linear(in_features: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return in_features @ weights.transpose(-1, -2)


def embedding(token_ids: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return weights[token_ids]


def silu(in_features: torch.Tensor) -> torch.Tensor:
    return in_features * torch.sigmoid(in_features)


def swiglu(
    in_features: torch.Tensor,
    w1_weight: torch.Tensor,
    w2_weight: torch.Tensor,
    w3_weight: torch.Tensor,
) -> torch.Tensor:
    return linear(silu(linear(in_features, w1_weight)) * linear(in_features, w3_weight), w2_weight)


def rmsnorm(
    in_features: torch.Tensor,
    weights: torch.Tensor,
    eps: float,
) -> torch.Tensor:
    original_dtype = in_features.dtype
    x = in_features.to(torch.float32)
    normalized = x * torch.rsqrt(torch.mean(x * x, dim=-1, keepdim=True) + eps)
    return (normalized.to(original_dtype) * weights).to(original_dtype)


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = query.shape[-1]
    scores = query @ key.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return softmax(scores, dim=-1) @ value


def rope(
    in_query_or_key: torch.Tensor,
    theta: float,
    token_positions: torch.Tensor,
) -> torch.Tensor:
    d_k = in_query_or_key.shape[-1]
    half = d_k // 2
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype
    inv_freq = theta ** (-torch.arange(0, half, device=device, dtype=torch.float32) * 2 / d_k)
    positions = token_positions.to(device=device, dtype=torch.float32)
    angles = positions[..., None] * inv_freq
    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _reshape_heads(x: torch.Tensor, num_heads: int) -> torch.Tensor:
    *prefix, sequence_length, d_model = x.shape
    d_head = d_model // num_heads
    return x.reshape(*prefix, sequence_length, num_heads, d_head).transpose(-3, -2)


def _merge_heads(x: torch.Tensor) -> torch.Tensor:
    *prefix, num_heads, sequence_length, d_head = x.shape
    return x.transpose(-3, -2).reshape(*prefix, sequence_length, num_heads * d_head)


def multihead_self_attention(
    in_features: torch.Tensor,
    num_heads: int,
    q_proj_weight: torch.Tensor,
    k_proj_weight: torch.Tensor,
    v_proj_weight: torch.Tensor,
    o_proj_weight: torch.Tensor,
    theta: float | None = None,
    token_positions: torch.Tensor | None = None,
) -> torch.Tensor:
    q = _reshape_heads(linear(in_features, q_proj_weight), num_heads)
    k = _reshape_heads(linear(in_features, k_proj_weight), num_heads)
    v = _reshape_heads(linear(in_features, v_proj_weight), num_heads)
    sequence_length = in_features.shape[-2]
    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=in_features.device)
        if token_positions.ndim == 1:
            for _ in range(q.ndim - 3):
                token_positions = token_positions.unsqueeze(0)
        token_positions = token_positions.unsqueeze(-2)
        q = rope(q, theta, token_positions)
        k = rope(k, theta, token_positions)
    causal = torch.tril(
        torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=in_features.device)
    )
    attended = scaled_dot_product_attention(q, k, v, causal)
    return linear(_merge_heads(attended), o_proj_weight)


def transformer_block(
    in_features: torch.Tensor,
    weights: dict[str, torch.Tensor],
    num_heads: int,
    theta: float,
) -> torch.Tensor:
    x = in_features
    attn_in = rmsnorm(x, weights["ln1.weight"], 1e-5)
    x = x + multihead_self_attention(
        attn_in,
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        theta=theta,
    )
    ffn_in = rmsnorm(x, weights["ln2.weight"], 1e-5)
    x = x + swiglu(
        ffn_in,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
    )
    return x


def transformer_lm(
    in_indices: torch.Tensor,
    weights: dict[str, torch.Tensor],
    num_layers: int,
    num_heads: int,
    rope_theta: float,
) -> torch.Tensor:
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {
            key.removeprefix(prefix): value
            for key, value in weights.items()
            if key.startswith(prefix)
        }
        x = transformer_block(x, block_weights, num_heads, rope_theta)
    x = rmsnorm(x, weights["ln_final.weight"], 1e-5)
    return linear(x, weights["lm_head.weight"])


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        std = math.sqrt(2 / (d_in + d_out))
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        torch.nn.init.normal_(self.weight, mean=0.0, std=1.0)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        return embedding(token_ids, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


class SiLUFFN(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))


class IdentityNorm(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None = None):
        super().__init__()
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        return multihead_self_attention(
            x,
            self.num_heads,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.output_proj.weight,
            theta=self.rope_theta,
            token_positions=token_positions,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        norm_mode: str = "pre",
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.norm_mode = norm_mode
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        self.ln1 = norm_cls(d_model) if use_rmsnorm else norm_cls()
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_theta if use_rope else None)
        self.ln2 = norm_cls(d_model) if use_rmsnorm else norm_cls()
        self.ffn = SwiGLU(d_model, d_ff) if ffn_type == "swiglu" else SiLUFFN(d_model, d_ff)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "post":
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x))
        return x + self.ffn(self.ln2(x))


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
        norm_mode: str = "pre",
        use_rmsnorm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope_theta,
                    norm_mode=norm_mode,
                    use_rmsnorm=use_rmsnorm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model) if use_rmsnorm else IdentityNorm()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if token_ids.shape[-1] > self.context_length:
            token_ids = token_ids[..., -self.context_length :]
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
