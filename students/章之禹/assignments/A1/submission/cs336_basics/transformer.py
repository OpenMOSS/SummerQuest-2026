"""A from-scratch causal Transformer language model for CS336 Assignment 1."""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, ModuleList

from cs336_basics.nn import Embedding, Linear, RMSNorm, SiLUFeedForward, SwiGLU, softmax


class RotaryPositionalEmbedding(Module):
    """Apply rotary positional embeddings to adjacent feature pairs."""

    cos: Tensor
    sin: Tensor

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

        pair_indices = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(f"expected a final dimension of {self.d_k}, got {x.shape[-1]}")
        if token_positions.shape[-1] != x.shape[-2]:
            raise ValueError("token_positions and x must have the same sequence length")
        if token_positions.numel() == 0:
            return x

        positions = token_positions.to(device=self.cos.device, dtype=torch.long)
        if positions.min().item() < 0 or positions.max().item() >= self.max_seq_len:
            raise ValueError(f"token positions must be in [0, {self.max_seq_len})")

        # Add singleton batch dimensions immediately before the sequence axis so
        # one position tensor can be shared across attention heads.
        while positions.ndim < x.ndim - 1:
            positions = positions.unsqueeze(-2)

        cos = self.cos[positions].to(dtype=x.dtype)
        sin = self.sin[positions].to(dtype=x.dtype)
        even = x[..., 0::2]
        odd = x[..., 1::2]

        # This is the rotation convention used by the assignment reference:
        # [x_even cos - x_odd sin, x_even sin + x_odd cos].
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(start_dim=-2)


def scaled_dot_product_attention(
    query: Tensor,
    key: Tensor,
    value: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary batch dimensions.

    Boolean masks follow the assignment convention: ``True`` permits attention
    and ``False`` blocks it.  A fully masked query produces a zero output.
    """

    if query.shape[-1] != key.shape[-1]:
        raise ValueError("query and key must have the same feature dimension")
    if key.shape[-2] != value.shape[-2]:
        raise ValueError("key and value must have the same sequence length")
    if query.shape[-1] == 0:
        raise ValueError("query/key feature dimension must be non-empty")

    scores = query @ key.transpose(-1, -2)
    scores = scores / math.sqrt(query.shape[-1])
    if mask is not None:
        if mask.dtype is not torch.bool:
            raise TypeError("attention mask must have boolean dtype")
        scores = scores.masked_fill(~mask, float("-inf"))

    probabilities = softmax(scores, dim=-1)
    return probabilities @ value


class CausalMultiHeadSelfAttention(Module):
    """Batched causal multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        use_rope: bool = True,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must both be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be evenly divisible by num_heads")
        if use_rope and max_seq_len is None:
            raise ValueError("max_seq_len is required when RoPE is enabled")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.max_seq_len = max_seq_len

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(theta, self.head_dim, max_seq_len, device=device)
            if use_rope and max_seq_len is not None
            else None
        )

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        split = x.reshape(*x.shape[:-2], sequence_length, self.num_heads, self.head_dim)
        return split.transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        merged = x.transpose(-3, -2)
        return merged.reshape(*merged.shape[:-3], sequence_length, self.d_model)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected a final dimension of {self.d_model}, got {x.shape[-1]}")

        sequence_length = x.shape[-2]
        if self.max_seq_len is not None and sequence_length > self.max_seq_len:
            raise ValueError(f"sequence length {sequence_length} exceeds maximum {self.max_seq_len}")

        query = self._split_heads(self.q_proj(x))
        key = self._split_heads(self.k_proj(x))
        value = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=x.device)
            query = self.rope(query, token_positions)
            key = self.rope(key, token_positions)

        causal_mask = torch.ones(sequence_length, sequence_length, device=x.device, dtype=torch.bool).tril()
        attended = scaled_dot_product_attention(query, key, value, causal_mask)
        return self.output_proj(self._merge_heads(attended))


# A concise public alias for callers that already know the module is causal.
MultiHeadSelfAttention = CausalMultiHeadSelfAttention


class TransformerBlock(Module):
    """A Transformer block supporting the assignment's architecture ablations."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str | None = None,
        silu_d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.remove_rmsnorm = remove_rmsnorm
        self.use_post_norm = use_post_norm

        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=not remove_rope,
            device=device,
            dtype=dtype,
        )

        normalized_ffn_type = "swiglu" if ffn_type is None else ffn_type.lower()
        if normalized_ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif normalized_ffn_type in {"silu", "ungated_silu"}:
            self.ffn = SiLUFeedForward(d_model, silu_d_ff, device=device, dtype=dtype)
        else:
            raise ValueError("ffn_type must be one of None, 'swiglu', 'silu', or 'ungated_silu'")

        if remove_rmsnorm:
            self.ln1 = None
            self.ln2 = None
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.use_post_norm:
            x = x + self.attn(x, token_positions)
            if self.ln1 is not None:
                x = self.ln1(x)
            x = x + self.ffn(x)
            if self.ln2 is not None:
                x = self.ln2(x)
            return x

        attention_input = self.ln1(x) if self.ln1 is not None else x
        x = x + self.attn(attention_input, token_positions)
        ffn_input = self.ln2(x) if self.ln2 is not None else x
        return x + self.ffn(ffn_input)


class TransformerLM(Module):
    """A decoder-only Transformer language model returning next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str | None = None,
        silu_d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must both be positive")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList(
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
                    silu_d_ff=silu_d_ff,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        # A final norm is part of pre-norm Transformers.  Post-norm blocks have
        # already normalized their residual stream, so they do not add one here.
        self.ln_final = (
            None if remove_rmsnorm or use_post_norm else RMSNorm(d_model, device=device, dtype=dtype)
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"sequence length {sequence_length} exceeds context length {self.context_length}")

        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=token_ids.device)

        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
