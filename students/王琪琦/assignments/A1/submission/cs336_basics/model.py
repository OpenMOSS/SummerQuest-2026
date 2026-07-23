from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from cs336_basics.nn_utils import softmax


def _truncated_normal_(tensor: Tensor, std: float) -> None:
    """Initialize from N(0, std**2), rejecting samples beyond three stddevs."""
    with torch.no_grad():
        tensor.normal_(mean=0.0, std=std)
        invalid = tensor.abs() > 3 * std
        while invalid.any():
            tensor[invalid] = torch.empty_like(tensor[invalid]).normal_(mean=0.0, std=std)
            invalid = tensor.abs() > 3 * std


class Linear(nn.Module):
    """A bias-free linear projection with weights stored as (d_out, d_in)."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("linear dimensions must be positive")
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        _truncated_normal_(self.weight, math.sqrt(2 / (d_in + d_out)))

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """A learned lookup table implemented without nn.Embedding."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("embedding dimensions must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(
            torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype)
        )
        _truncated_normal_(self.weight, 1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
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

    def forward(self, inputs: Tensor) -> Tensor:
        input_dtype = inputs.dtype
        inputs_float = inputs.to(torch.float32)
        normalized = inputs_float * torch.rsqrt(inputs_float.square().mean(dim=-1, keepdim=True) + self.eps)
        return normalized.to(input_dtype) * self.weight


def silu(inputs: Tensor) -> Tensor:
    return inputs * torch.sigmoid(inputs)


class SwiGLU(nn.Module):
    """SwiGLU feed-forward network: W2(SiLU(W1(x)) * W3(x))."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)) * self.w3(inputs))


class SiLUFeedForward(nn.Module):
    """Two-projection SiLU FFN used by the matched-parameter ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)))


class RotaryPositionalEmbedding(nn.Module):
    """RoPE using adjacent even/odd feature pairs."""

    cos: Tensor
    sin: Tensor

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        *,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("d_k must be a positive even number")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive")
        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        frequency_exponents = torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        inverse_frequencies = theta**-frequency_exponents
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.register_buffer("cos", angles.cos(), persistent=False)
        self.register_buffer("sin", angles.sin(), persistent=False)

    def forward(self, inputs: Tensor, token_positions: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_k:
            raise ValueError(f"expected final dimension {self.d_k}, got {inputs.shape[-1]}")
        if token_positions.numel() and (
            token_positions.min() < 0 or token_positions.max() >= self.max_seq_len
        ):
            raise ValueError("token position is outside the precomputed RoPE range")

        cos = self.cos[token_positions].to(dtype=inputs.dtype)
        sin = self.sin[token_positions].to(dtype=inputs.dtype)
        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos
        return torch.stack((rotated_even, rotated_odd), dim=-1).flatten(-2)


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over the final two dimensions."""
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("queries and keys must have the same feature dimension")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("keys and values must have the same sequence length")

    scores = queries @ keys.transpose(-1, -2)
    scores = scores / math.sqrt(queries.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must have boolean dtype")
        scores = scores.masked_fill(~mask, -torch.inf)
        row_max = scores.max(dim=-1, keepdim=True).values
        row_max = torch.where(torch.isfinite(row_max), row_max, torch.zeros_like(row_max))
        exponentials = torch.where(mask, torch.exp(scores - row_max), torch.zeros_like(scores))
        denominator = exponentials.sum(dim=-1, keepdim=True)
        weights = torch.where(
            denominator > 0,
            exponentials / denominator.clamp_min(torch.finfo(scores.dtype).tiny),
            torch.zeros_like(exponentials),
        )
    else:
        weights = softmax(scores, dim=-1)
    return weights @ values


class MultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional rotary positions."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        rope_theta: float | None = None,
        max_seq_len: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads")
        if (rope_theta is None) != (max_seq_len is None):
            raise ValueError("rope_theta and max_seq_len must be provided together")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = (
            RotaryPositionalEmbedding(rope_theta, self.d_head, max_seq_len, device=device)
            if rope_theta is not None and max_seq_len is not None
            else None
        )

    def _split_heads(self, inputs: Tensor) -> Tensor:
        return inputs.unflatten(-1, (self.num_heads, self.d_head)).transpose(-3, -2)

    def _merge_heads(self, inputs: Tensor) -> Tensor:
        return inputs.transpose(-3, -2).flatten(-2)

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = inputs.shape[-2]
        queries = self._split_heads(self.q_proj(inputs))
        keys = self._split_heads(self.k_proj(inputs))
        values = self._split_heads(self.v_proj(inputs))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=inputs.device)
            # Insert the head axis so batch-specific positions broadcast over heads.
            positions_with_head_axis = token_positions.unsqueeze(-2)
            queries = self.rope(queries, positions_with_head_axis)
            keys = self.rope(keys, positions_with_head_axis)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=inputs.device,
        ).tril()
        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._merge_heads(attended))


class TransformerBlock(nn.Module):
    """A configurable Transformer block supporting the required ablations."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        rope_theta: float,
        *,
        eps: float = 1e-5,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if norm_position not in {"pre", "post"}:
            raise ValueError("norm_position must be 'pre' or 'post'")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'")
        self.norm_position = norm_position
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            rope_theta=rope_theta if use_rope else None,
            max_seq_len=max_seq_len if use_rope else None,
            device=device,
            dtype=dtype,
        )
        ffn_class = SwiGLU if ffn_type == "swiglu" else SiLUFeedForward
        self.ffn = ffn_class(d_model, d_ff, device=device, dtype=dtype)
        if use_rmsnorm:
            self.ln1 = RMSNorm(d_model, eps, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, eps, device=device, dtype=dtype)
        else:
            self.ln1 = nn.Identity()
            self.ln2 = nn.Identity()

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_position == "post":
            residual = self.ln1(inputs + self.attn(inputs, token_positions))
            return self.ln2(residual + self.ffn(residual))
        residual = inputs + self.attn(self.ln1(inputs), token_positions)
        return residual + self.ffn(self.ln2(residual))


class TransformerLM(nn.Module):
    """A decoder-only Transformer language model that returns unnormalized logits."""

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
        eps: float = 1e-5,
        use_rmsnorm: bool = True,
        norm_position: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive")
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers

        self.token_embeddings = Embedding(
            vocab_size,
            d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    eps=eps,
                    use_rmsnorm=use_rmsnorm,
                    norm_position=norm_position,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = (
            RMSNorm(d_model, eps, device=device, dtype=dtype) if use_rmsnorm else nn.Identity()
        )
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"sequence length {sequence_length} exceeds context length {self.context_length}"
            )

        hidden = self.token_embeddings(token_ids)
        positions = torch.arange(sequence_length, device=token_ids.device)
        for layer in self.layers:
            hidden = layer(hidden, positions)
        return self.lm_head(self.ln_final(hidden))
