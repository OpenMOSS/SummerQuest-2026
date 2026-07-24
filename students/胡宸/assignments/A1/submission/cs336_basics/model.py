from __future__ import annotations

import math

import torch
from torch import Tensor
from torch import nn


def linear(x: Tensor, weight: Tensor) -> Tensor:
    return x @ weight.transpose(-1, -2)


def embedding(token_ids: Tensor, weight: Tensor) -> Tensor:
    return weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


def rmsnorm(x: Tensor, weight: Tensor, eps: float = 1e-5) -> Tensor:
    original_dtype = x.dtype
    x_float = x.float()
    normalized = x_float * torch.rsqrt(torch.mean(x_float * x_float, dim=-1, keepdim=True) + eps)
    return (normalized.to(original_dtype) * weight).to(original_dtype)


def swiglu(x: Tensor, w1_weight: Tensor, w2_weight: Tensor, w3_weight: Tensor) -> Tensor:
    return linear(silu(linear(x, w1_weight)) * linear(x, w3_weight), w2_weight)


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    shifted = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_z = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + torch.max(logits, dim=-1).values
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_z - target_logits)


def scaled_dot_product_attention(q: Tensor, k: Tensor, v: Tensor, mask: Tensor | None = None) -> Tensor:
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, torch.finfo(scores.dtype).min)
    return softmax(scores, dim=-1) @ v


def rope(x: Tensor, theta: float, max_seq_len: int, token_positions: Tensor) -> Tensor:
    del max_seq_len
    d_k = x.shape[-1]
    if d_k % 2 != 0:
        raise ValueError("RoPE requires an even embedding dimension")

    half = d_k // 2
    idx = torch.arange(half, device=x.device, dtype=torch.float32)
    inv_freq = theta ** (-2 * idx / d_k)
    angles = token_positions.to(device=x.device, dtype=torch.float32).unsqueeze(-1) * inv_freq
    cos = torch.cos(angles).to(dtype=x.dtype)
    sin = torch.sin(angles).to(dtype=x.dtype)
    while cos.ndim < x.ndim:
        cos = cos.unsqueeze(-3)
        sin = sin.unsqueeze(-3)

    x_even = x[..., 0::2]
    x_odd = x[..., 1::2]
    out = torch.empty_like(x)
    out[..., 0::2] = x_even * cos - x_odd * sin
    out[..., 1::2] = x_even * sin + x_odd * cos
    return out


def _split_heads(x: Tensor, num_heads: int) -> Tensor:
    head_dim = x.shape[-1] // num_heads
    return x.reshape(*x.shape[:-1], num_heads, head_dim).transpose(-3, -2)


def _combine_heads(x: Tensor) -> Tensor:
    return x.transpose(-3, -2).reshape(*x.shape[:-3], x.shape[-2], x.shape[-3] * x.shape[-1])


def multihead_self_attention(
    x: Tensor,
    num_heads: int,
    q_proj_weight: Tensor,
    k_proj_weight: Tensor,
    v_proj_weight: Tensor,
    o_proj_weight: Tensor,
    *,
    theta: float | None = None,
    max_seq_len: int | None = None,
    token_positions: Tensor | None = None,
) -> Tensor:
    seq_len = x.shape[-2]
    q = _split_heads(linear(x, q_proj_weight), num_heads)
    k = _split_heads(linear(x, k_proj_weight), num_heads)
    v = _split_heads(linear(x, v_proj_weight), num_heads)

    if theta is not None:
        if max_seq_len is None:
            max_seq_len = seq_len
        if token_positions is None:
            token_positions = torch.arange(seq_len, device=x.device)
        q = rope(q, theta, max_seq_len, token_positions)
        k = rope(k, theta, max_seq_len, token_positions)

    causal_mask = torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=x.device))
    y = scaled_dot_product_attention(q, k, v, causal_mask)
    return linear(_combine_heads(y), o_proj_weight)


def transformer_block(
    x: Tensor,
    num_heads: int,
    max_seq_len: int,
    theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    attn_in = rmsnorm(x, weights["ln1.weight"])
    x = x + multihead_self_attention(
        attn_in,
        num_heads,
        weights["attn.q_proj.weight"],
        weights["attn.k_proj.weight"],
        weights["attn.v_proj.weight"],
        weights["attn.output_proj.weight"],
        theta=theta,
        max_seq_len=max_seq_len,
    )
    ffn_in = rmsnorm(x, weights["ln2.weight"])
    return x + swiglu(ffn_in, weights["ffn.w1.weight"], weights["ffn.w2.weight"], weights["ffn.w3.weight"])


def transformer_lm(
    in_indices: Tensor,
    num_layers: int,
    num_heads: int,
    context_length: int,
    rope_theta: float,
    weights: dict[str, Tensor],
) -> Tensor:
    x = embedding(in_indices, weights["token_embeddings.weight"])
    for layer_idx in range(num_layers):
        prefix = f"layers.{layer_idx}."
        block_weights = {
            key.removeprefix(prefix): value for key, value in weights.items() if key.startswith(prefix)
        }
        x = transformer_block(x, num_heads, context_length, rope_theta, block_weights)
    x = rmsnorm(x, weights["ln_final.weight"])
    return linear(x, weights["lm_head.weight"])


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=math.sqrt(2 / (d_in + d_out)), a=-3.0, b=3.0)

    def forward(self, x: Tensor) -> Tensor:
        return linear(x, self.weight)


class Embedding(nn.Module):
    def __init__(self, vocab_size: int, d_model: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(vocab_size, d_model))
        torch.nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return embedding(token_ids, self.weight)


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x: Tensor) -> Tensor:
        return rmsnorm(x, self.weight, self.eps)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)
        self.w3 = Linear(d_model, d_ff)

    def forward(self, x: Tensor) -> Tensor:
        return swiglu(x, self.w1.weight, self.w2.weight, self.w3.weight)


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int):
        super().__init__()
        self.w1 = Linear(d_model, d_ff)
        self.w2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class IdentityNorm(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class MultiheadSelfAttention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, max_seq_len: int, theta: float, use_rope: bool = True):
        super().__init__()
        self.num_heads = num_heads
        self.max_seq_len = max_seq_len
        self.theta = theta if use_rope else None
        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.output_proj = Linear(d_model, d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        return multihead_self_attention(
            x,
            self.num_heads,
            self.q_proj.weight,
            self.k_proj.weight,
            self.v_proj.weight,
            self.output_proj.weight,
            theta=self.theta,
            max_seq_len=self.max_seq_len,
            token_positions=token_positions,
        )


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        silu_d_ff: int | None = None,
    ):
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError(f"norm_position must be 'pre' or 'post', got {norm_position!r}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type!r}")

        self.norm_position = norm_position
        self.attn = MultiheadSelfAttention(d_model, num_heads, max_seq_len, theta, use_rope=use_rope)
        norm_cls = RMSNorm if use_rmsnorm else IdentityNorm
        self.ln1 = norm_cls(d_model) if use_rmsnorm else norm_cls()
        self.ln2 = norm_cls(d_model) if use_rmsnorm else norm_cls()
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        else:
            self.ffn = SiLUFeedForward(d_model, silu_d_ff if silu_d_ff is not None else d_ff)

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
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        silu_d_ff: int | None = None,
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
                    context_length,
                    rope_theta,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    silu_d_ff=silu_d_ff,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model) if use_rmsnorm else IdentityNorm()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.shape[-1] > self.context_length:
            token_ids = token_ids[..., -self.context_length :]
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x)
        return self.lm_head(self.ln_final(x))
