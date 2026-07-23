"""Transformer model components used throughout assignment 1.

The implementations in this module intentionally build the core operations from
PyTorch tensor primitives rather than delegating to the corresponding high-level
``torch.nn`` or ``torch.nn.functional`` layers.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU (Swish) activation."""

    return x * torch.sigmoid(x)


class Linear(nn.Module):
    """A bias-free linear transformation."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (in_features + out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """Map integer token IDs to learned embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root mean square layer normalization with a learned gain."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms_inverse = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = x_float * rms_inverse
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)


class SwiGLU(nn.Module):
    """A bias-free SwiGLU position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_ff is None:
            d_ff = max(64, 64 * round((8.0 * d_model / 3.0) / 64.0))
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFFN(nn.Module):
    """A bias-free SiLU position-wise feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))



class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary position embeddings to adjacent feature pairs."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if theta <= 0:
            raise ValueError("theta must be positive")
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even integer")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos_cached", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cached", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected the final dimension to be {self.d_k}, got {x.shape[-1]}")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions and x must have the same sequence length")

        cos_cached = self.get_buffer("cos_cached")
        sin_cached = self.get_buffer("sin_cached")
        token_positions = token_positions.to(device=cos_cached.device, dtype=torch.long)
        if token_positions.numel() > 0:
            min_position = int(token_positions.min().item())
            max_position = int(token_positions.max().item())
            if min_position < 0 or max_position >= self.max_seq_len:
                raise ValueError(f"token positions must be in [0, {self.max_seq_len})")

        cos = cos_cached[token_positions]
        sin = sin_cached[token_positions]
        while cos.ndim < x.ndim:
            # Insert omitted batch-like dimensions immediately before sequence.
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)
        cos = cos.to(device=x.device, dtype=x.dtype)
        sin = sin.to(device=x.device, dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def _stable_softmax(x: Tensor, dim: int = -1) -> Tensor:
    """Numerically stable softmax, including a defined all-masked result."""

    maxima = x.amax(dim=dim, keepdim=True)
    finite_maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    exponentials = torch.exp(x - finite_maxima)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    safe_denominator = denominator.clamp_min(torch.finfo(exponentials.dtype).tiny)
    return exponentials / safe_denominator


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary batch dimensions."""

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same feature dimension")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same number of key positions")

    scores = (Q @ K.transpose(-1, -2)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError("attention mask must have boolean dtype")
        mask = mask.to(device=scores.device)
        scores = scores.masked_fill(~mask, -torch.inf)

    probabilities = _stable_softmax(scores, dim=-1)
    return probabilities @ V


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        causal: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.causal = causal
        self.rope = rope
        if rope is not None and rope.d_k != self.d_head:
            raise ValueError("RoPE dimension must equal d_model // num_heads")

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        leading_shape = x.shape[:-2]
        return x.reshape(*leading_shape, sequence_length, self.num_heads, self.d_head).transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        leading_shape = x.shape[:-3]
        return x.transpose(-3, -2).contiguous().reshape(*leading_shape, sequence_length, self.d_model)

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        mask: Tensor | None = None,
        causal: bool | None = None,
    ) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected the final dimension to be {self.d_model}, got {x.shape[-1]}")

        sequence_length = x.shape[-2]
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        attention_mask = mask
        if attention_mask is not None:
            if attention_mask.dtype is not torch.bool:
                raise TypeError("attention mask must have boolean dtype")
            while attention_mask.ndim < q.ndim:
                attention_mask = attention_mask.unsqueeze(-3)

        use_causal_mask = self.causal if causal is None else causal
        if use_causal_mask:
            causal_mask = torch.ones(
                sequence_length,
                sequence_length,
                dtype=torch.bool,
                device=x.device,
            ).tril()
            attention_mask = causal_mask if attention_mask is None else attention_mask.to(x.device) & causal_mask

        attended = scaled_dot_product_attention(q, k, v, mask=attention_mask)
        return self.output_proj(self._merge_heads(attended))


class TransformerBlock(nn.Module):
    """A Transformer block with configurable normalization and FFN."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        rope: RotaryPositionalEmbedding | None = None,
        causal: bool = True,
        normalization: str = "rmsnorm",
        ffn_type: str = "swiglu",
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if rope is not None and max_seq_len is not None:
            raise ValueError("pass either rope or max_seq_len, not both")
        if normalization not in {"rmsnorm", "none"}:
            raise ValueError("normalization must be either 'rmsnorm' or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be either 'swiglu' or 'silu'")

        self.normalization = normalization
        self.ffn_type = ffn_type

        if rope is None and max_seq_len is not None:
            rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=rope,
            causal=causal,
            device=device,
            dtype=dtype,
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if normalization == "rmsnorm" else nn.Identity()
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFFN(d_model, d_ff, device=device, dtype=dtype)
        )
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if normalization == "rmsnorm" else nn.Identity()

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))

class TransformerLM(nn.Module):
    """A decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        normalization: str = "rmsnorm",
        position_encoding: str = "rope",
        ffn_type: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0 or num_layers <= 0 or num_heads <= 0 or d_ff <= 0:
            raise ValueError("all model dimensions and counts must be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")
        if normalization not in {"rmsnorm", "none"}:
            raise ValueError("normalization must be either 'rmsnorm' or 'none'")
        if position_encoding not in {"rope", "none"}:
            raise ValueError("position_encoding must be either 'rope' or 'none'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be either 'swiglu' or 'silu'")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.normalization = normalization
        self.position_encoding = position_encoding
        self.ffn_type = ffn_type

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        rope = (
            RotaryPositionalEmbedding(
                rope_theta,
                d_model // num_heads,
                context_length,
                device=device,
            )
            if position_encoding == "rope"
            else None
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    device=device,
                    dtype=dtype,
                    rope=rope,
                    causal=True,
                    normalization=normalization,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if normalization == "rmsnorm" else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, in_indices: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = in_indices.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"input sequence length {sequence_length} exceeds context length {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=in_indices.device)

        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))
# A descriptive alias used by some assignment solutions.
CausalMultiHeadSelfAttention = MultiHeadSelfAttention


__all__ = [
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RotaryPositionalEmbedding",
    "SiLUFFN",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "scaled_dot_product_attention",
    "silu",
]
