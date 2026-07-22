from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _initialize_weight(weight: Tensor, std: float) -> None:
    with torch.no_grad():
        nn.init.trunc_normal_(weight, mean=0.0, std=std, a=-3 * std, b=3 * std)


class Identity(nn.Module):
    def forward(self, x: Tensor) -> Tensor:
        return x


class Linear(nn.Module):
    def __init__(self, d_in: int, d_out: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        _initialize_weight(self.weight, math.sqrt(2 / (d_in + d_out)))

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    def __init__(self, num_embeddings: int, embedding_dim: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _initialize_weight(self.weight, 1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *, device=None, dtype=None) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RMSNorm(nn.Module):
    def __init__(self, d_model: int, eps: float = 1e-5, *, device=None, dtype=None) -> None:
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        rms = torch.sqrt(torch.mean(x_float.square(), dim=-1, keepdim=True) + self.eps)
        normalized = (x_float / rms).to(input_dtype)
        return normalized * self.weight


class RotaryPositionalEmbedding(nn.Module):
    def __init__(self, theta: float, d_k: int, max_seq_len: int, *, device=None) -> None:
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE dimension must be even")

        frequencies = theta ** (-torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = torch.outer(positions, frequencies)
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)
        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        output = torch.empty_like(x)
        output[..., 0::2] = x_even * cos - x_odd * sin
        output[..., 1::2] = x_even * sin + x_odd * cos
        return output


def softmax(x: Tensor, dim: int) -> Tensor:
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    scores = queries @ keys.transpose(-1, -2) / math.sqrt(queries.shape[-1])
    if mask is not None:
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = softmax(scores, dim=-1)
    return probabilities @ values


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        split = x.reshape(*x.shape[:-1], self.num_heads, self.head_dim)
        return split.transpose(-3, -2)

    def _combine_heads(self, x: Tensor) -> Tensor:
        combined = x.transpose(-3, -2).contiguous()
        return combined.reshape(*combined.shape[:-2], self.num_heads * self.head_dim)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(x.shape[-2], device=x.device)
            elif token_positions.ndim == queries.ndim - 2:
                token_positions = token_positions.unsqueeze(-2)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        sequence_length = x.shape[-2]
        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=x.device,
        ).tril()
        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._combine_heads(attended))


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.use_post_norm = use_post_norm
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=None if remove_rope else theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        elif ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            raise ValueError(f"unknown ffn_type: {ffn_type}")
        self.ln1 = Identity() if remove_rmsnorm else RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.ln2 = Identity() if remove_rmsnorm else RMSNorm(d_model=d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.use_post_norm:
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions)
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
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    remove_rmsnorm=remove_rmsnorm,
                    use_post_norm=use_post_norm,
                    remove_rope=remove_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = Identity() if remove_rmsnorm else RMSNorm(d_model=d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.shape[-1] > self.context_length:
            raise ValueError("input sequence exceeds context length")
        positions = torch.arange(token_ids.shape[-1], device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        return self.lm_head(self.ln_final(x))
