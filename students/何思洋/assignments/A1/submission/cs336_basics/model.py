from __future__ import annotations

import math

import torch
from torch import nn
from torch import Tensor


def linear(in_features: Tensor, weights: Tensor) -> Tensor:
    return in_features @ weights.transpose(-1, -2)


def embedding(token_ids: Tensor, weights: Tensor) -> Tensor:
    return weights[token_ids]


def silu(in_features: Tensor) -> Tensor:
    return in_features * torch.sigmoid(in_features)


def rmsnorm(in_features: Tensor, weights: Tensor, eps: float = 1e-5) -> Tensor:
    original_dtype = in_features.dtype
    normalized = in_features.float()
    rms = torch.sqrt(torch.mean(normalized * normalized, dim=-1, keepdim=True) + eps)
    return (normalized / rms * weights).to(original_dtype)


def swiglu(in_features: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    return linear(silu(linear(in_features, w1_weight)) * linear(in_features, w3_weight), w2_weight)


def softmax(in_features: Tensor, dim: int) -> Tensor:
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    d_k = Q.shape[-1]
    scores = Q @ K.transpose(-1, -2) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return softmax(scores, dim=-1) @ V


def rope(
    in_query_or_key: Tensor,
    theta: float,
    token_positions: Tensor,
) -> Tensor:
    d_k = in_query_or_key.shape[-1]
    half = d_k // 2
    device = in_query_or_key.device
    dtype = in_query_or_key.dtype

    freq_indices = torch.arange(half, device=device, dtype=torch.float32)
    inv_freq = theta ** (-2 * freq_indices / d_k)
    angles = token_positions.to(device=device, dtype=torch.float32).unsqueeze(-1) * inv_freq

    cos = torch.cos(angles).to(dtype)
    sin = torch.sin(angles).to(dtype)
    while cos.ndim < in_query_or_key.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    x_even = in_query_or_key[..., 0::2]
    x_odd = in_query_or_key[..., 1::2]
    out = torch.empty_like(in_query_or_key)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _causal_mask(sequence_length: int, device: torch.device) -> Tensor:
    return torch.tril(torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=device))


def multihead_self_attention(
    in_features: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    theta: float | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    d_model = in_features.shape[-1]
    d_head = d_model // num_heads
    sequence_length = in_features.shape[-2]

    q = linear(in_features, q_proj_weight)
    k = linear(in_features, k_proj_weight)
    v = linear(in_features, v_proj_weight)

    q = q.unflatten(-1, (num_heads, d_head)).transpose(-2, -3)
    k = k.unflatten(-1, (num_heads, d_head)).transpose(-2, -3)
    v = v.unflatten(-1, (num_heads, d_head)).transpose(-2, -3)

    if theta is not None:
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=in_features.device)
        q = rope(q, theta=theta, token_positions=token_positions)
        k = rope(k, theta=theta, token_positions=token_positions)

    mask = _causal_mask(sequence_length, in_features.device)
    attended = scaled_dot_product_attention(q, k, v, mask=mask)
    attended = attended.transpose(-2, -3).flatten(-2)
    return linear(attended, o_proj_weight)


def transformer_block(
    in_features: Tensor,
    num_heads: int,
    theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    attn_input = rmsnorm(in_features, weights["ln1.weight"])
    hidden = in_features + multihead_self_attention(
        attn_input,
        num_heads=num_heads,
        q_proj_weight=weights["attn.q_proj.weight"],
        k_proj_weight=weights["attn.k_proj.weight"],
        v_proj_weight=weights["attn.v_proj.weight"],
        o_proj_weight=weights["attn.output_proj.weight"],
        theta=theta,
    )
    ffn_input = rmsnorm(hidden, weights["ln2.weight"])
    return hidden + swiglu(
        ffn_input,
        weights["ffn.w1.weight"],
        weights["ffn.w2.weight"],
        weights["ffn.w3.weight"],
    )


def transformer_lm(
    in_indices: Tensor,
    num_layers: int,
    num_heads: int,
    rope_theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    hidden = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        layer_weights = {
            key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)
        }
        hidden = transformer_block(hidden, num_heads=num_heads, theta=rope_theta, weights=layer_weights)
    hidden = rmsnorm(hidden, weights["ln_final.weight"])
    return linear(hidden, weights["lm_head.weight"])


def _normal_parameter(*shape: int, std: float = 0.02) -> nn.Parameter:
    return nn.Parameter(torch.randn(*shape) * std)


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int) -> None:
        super().__init__()
        self.weight = _normal_parameter(d_out, d_in, std=math.sqrt(2 / (d_in + d_out)))

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int) -> None:
        super().__init__()
        self.weight = _normal_parameter(vocab_size, d_model, std=0.02)

    def forward(self, token_ids: Tensor) -> Tensor:
        return embedding(token_ids, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, eps=self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, activation: str = "swiglu") -> None:
        super().__init__()
        if activation not in {"swiglu", "linear-gate"}:
            raise ValueError(f"unsupported FFN activation: {activation}")
        self.activation = activation
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        if self.activation == "swiglu":
            return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)
        return linear(linear(x, self.w1.weight) * linear(x, self.w3.weight), self.w2.weight)


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, rope_theta: float | None = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.rope_theta = rope_theta
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return multihead_self_attention(
            x,
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            o_proj_weight=self.output_proj.weight,
            theta=self.rope_theta,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float | None,
        norm_mode: str = "pre",
        ffn_activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if norm_mode not in {"pre", "post", "none"}:
            raise ValueError(f"unsupported norm mode: {norm_mode}")
        self.norm_mode = norm_mode
        self.ln1 = RMSNorm(d_model)
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope_theta=rope_theta)
        self.ln2 = RMSNorm(d_model)
        self.ffn = SwiGLU(d_model, d_ff, activation=ffn_activation)

    def forward(self, x: Tensor) -> Tensor:
        if self.norm_mode == "pre":
            x = x + self.attn(self.ln1(x))
            return x + self.ffn(self.ln2(x))
        if self.norm_mode == "post":
            x = self.ln1(x + self.attn(x))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x)
        return x + self.ffn(x)


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
        use_rope: bool = True,
        ffn_activation: str = "swiglu",
    ) -> None:
        super().__init__()
        if norm_mode not in {"pre", "post", "none"}:
            raise ValueError(f"unsupported norm mode: {norm_mode}")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.norm_mode = norm_mode
        self.token_embeddings = Embedding(vocab_size, d_model)
        block_rope_theta = rope_theta if use_rope else None
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    rope_theta=block_rope_theta,
                    norm_mode=norm_mode,
                    ffn_activation=ffn_activation,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model)
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        hidden = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden = layer(hidden)
        if self.norm_mode != "none":
            hidden = self.ln_final(hidden)
        return self.lm_head(hidden)
