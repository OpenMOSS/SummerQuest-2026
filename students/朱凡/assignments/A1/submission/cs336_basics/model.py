"""Transformer language-model components used throughout assignment 1.

The implementations deliberately expose the individual building blocks instead of
wrapping PyTorch's high-level Transformer modules.  This keeps the parameter names
and tensor operations aligned with the notation in the assignment handout.
"""

from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """A bias-free linear transformation ``y = W x``."""

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
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum("...i,oi->...o", x, self.weight)


class Embedding(nn.Module):
    """A trainable token lookup table."""

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
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("token_ids must be an integer tensor")
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization with a learned elementwise gain."""

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
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.float()
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * inverse_rms).to(input_dtype)
        return normalized * self.weight.to(input_dtype)


def silu(x: Tensor) -> Tensor:
    """The SiLU activation, written explicitly as required by the assignment."""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            # Round (8 / 3) * d_model to the nearest multiple of 64.
            d_ff = max(64, 64 * round((8 * d_model / 3) / 64))
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Ungated SiLU feed-forward baseline used by the section 7 ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 4 * d_model
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


class RotaryPositionEmbedding(nn.Module):
    """Rotary position embeddings (RoPE) for interleaved feature pairs."""

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
            raise ValueError("d_k must be a positive even number")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(0, d_k, 2, dtype=torch.float32, device=device)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, dtype=torch.float32, device=device)
        angles = torch.outer(positions, inverse_frequencies)
        # Caches are derived constants, not model parameters or checkpoint state.
        self.cos_cached: Tensor
        self.sin_cached: Tensor
        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected final dimension {self.d_k}, got {x.shape[-1]}")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions and x must have the same sequence length")
        if token_positions.numel() and (
            token_positions.min().item() < 0 or token_positions.max().item() >= self.max_seq_len
        ):
            raise ValueError("token position is outside the configured context length")

        positions = token_positions.to(device=self.cos_cached.device, dtype=torch.long)
        cos = self.cos_cached[positions]
        sin = self.sin_cached[positions]
        while cos.ndim < x.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        x_pairs = x.reshape(*x.shape[:-1], self.d_k // 2, 2)
        x_even, x_odd = x_pairs.unbind(dim=-1)
        cos = cos.to(dtype=x.dtype)
        sin = sin.to(dtype=x.dtype)
        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


# Short alias commonly used in assignment solutions.
RoPE = RotaryPositionEmbedding


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim``."""

    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = shifted.exp()
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def scaled_dot_product_attention(Q: Tensor, K: Tensor, V: Tensor, mask: Tensor | None = None) -> Tensor:
    """Compute scaled dot-product attention over the penultimate dimension."""

    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same key dimension")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same number of key positions")

    scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must be boolean")
        scores = scores.masked_fill(~mask, -torch.inf)
    attention_weights = softmax(scores, dim=-1)
    return torch.einsum("...qk,...kv->...qv", attention_weights, V)


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if (max_seq_len is None) != (theta is None):
            raise ValueError("max_seq_len and theta must either both be set or both be omitted")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        *leading, sequence_length, _ = x.shape
        return x.reshape(*leading, sequence_length, self.num_heads, self.head_dim).transpose(-3, -2)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.ones(sequence_length, sequence_length, dtype=torch.bool, device=x.device).tril()
        attended = scaled_dot_product_attention(q, k, v, causal_mask)
        merged = attended.transpose(-3, -2).contiguous().reshape(*x.shape[:-2], sequence_length, self.d_model)
        return self.output_proj(merged)


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        norm_mode: Literal["pre", "post", "none"] = "pre",
        use_rope: bool = True,
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
    ) -> None:
        super().__init__()
        if norm_mode not in ("pre", "post", "none"):
            raise ValueError("norm_mode must be 'pre', 'post', or 'none'")
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_mode = norm_mode
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len if use_rope else None,
            theta=theta if use_rope else None,
            device=device,
            dtype=dtype,
        )
        if norm_mode == "none":
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_mode == "post":
            x = self.ln1(x + self.attn(x, token_positions=token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions=token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        norm_mode: Literal["pre", "post", "none"] = "pre",
        use_rope: bool = True,
        ffn_type: Literal["swiglu", "silu"] = "swiglu",
        tied_embeddings: bool = False,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    device=device,
                    dtype=dtype,
                    norm_mode=norm_mode,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = nn.Identity() if norm_mode == "none" else RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        if tied_embeddings:
            # Keep the LM head's small initialization scale. Pointing the head at
            # the default Embedding weight would inherit its std=1 initialization
            # and produce extremely large initial logits.
            self.token_embeddings.weight = self.lm_head.weight

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"input sequence length {sequence_length} exceeds context length {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=token_ids.device)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        return self.lm_head(self.ln_final(x))


# Compatibility aliases for descriptive names used in some notebooks/scripts.
CausalMultiHeadSelfAttention = MultiHeadSelfAttention
TransformerLanguageModel = TransformerLM
