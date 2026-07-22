"""Core Transformer language-model components used throughout assignment 1.

The modules in this file intentionally implement the primitive operations directly
instead of delegating to ``torch.nn.Linear``, ``torch.nn.Embedding``, or PyTorch's
fused attention implementation.  Besides making the underlying computations
explicit, this keeps the parameter names compatible with the reference checkpoints
used by the assignment tests.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """A bias-free linear transformation over the final input dimension."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must both be positive")

        self.in_features = int(in_features)
        self.out_features = int(out_features)
        self.weight = nn.Parameter(torch.empty((self.out_features, self.in_features), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or x.shape[-1] != self.in_features:
            raise ValueError(f"expected the final input dimension to be {self.in_features}, got shape {tuple(x.shape)}")
        return torch.matmul(x, self.weight.transpose(0, 1))

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias=False"


class Embedding(nn.Module):
    """A learnable lookup table mapping token IDs to embedding vectors."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must both be positive")

        self.num_embeddings = int(num_embeddings)
        self.embedding_dim = int(embedding_dim)
        self.weight = nn.Parameter(torch.empty((self.num_embeddings, self.embedding_dim), device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]

    def extra_repr(self) -> str:
        return f"num_embeddings={self.num_embeddings}, embedding_dim={self.embedding_dim}"


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

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

        self.d_model = int(d_model)
        self.eps = float(eps)
        self.weight = nn.Parameter(torch.ones(self.d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim == 0 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected shape (..., {self.d_model}), got {tuple(x.shape)}")

        # Squaring fp16/bfloat16 values can overflow even when the normalized result
        # itself is representable, so the assignment explicitly requires fp32 here.
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        inverse_rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = x_float * inverse_rms * self.weight.to(torch.float32)
        return normalized.to(input_dtype)


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU (also called Swish) activation elementwise."""

    return x * torch.sigmoid(x)


class SiLU(nn.Module):
    """Module wrapper for :func:`silu`."""

    def forward(self, x: Tensor) -> Tensor:
        return silu(x)


class Identity(nn.Module):
    """Parameter-free identity used by architecture ablations."""

    def forward(self, x: Tensor) -> Tensor:
        return x


class SwiGLU(nn.Module):
    """The bias-free SwiGLU position-wise feed-forward network."""

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
            # The canonical width is 8/3 d_model.  A nearby multiple of 64 is
            # friendlier to accelerator kernels (512 -> 1344, for example).
            d_ff = max(64, round((8.0 * d_model / 3.0) / 64.0) * 64)
        if d_ff <= 0:
            raise ValueError("d_ff must be positive")

        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)
        self.w3 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """A conventional two-matrix, bias-free SiLU feed-forward network."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must both be positive")
        self.d_model = int(d_model)
        self.d_ff = int(d_ff)
        self.w1 = Linear(self.d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, self.d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along ``dim``.

    Rows containing only ``-inf`` arise naturally from a fully masked attention
    query.  Returning zeros for those rows makes masked attention well-defined and
    avoids propagating NaNs; ordinary finite rows are the standard stable softmax.
    """

    if x.ndim == 0:
        raise ValueError("softmax requires a tensor with at least one dimension")
    dim = dim if dim >= 0 else x.ndim + dim
    if dim < 0 or dim >= x.ndim:
        raise IndexError(f"dimension {dim} is out of range for a {x.ndim}D tensor")

    input_dtype = x.dtype
    working = x.to(torch.float32) if input_dtype in (torch.float16, torch.bfloat16) else x
    maximum = torch.amax(working, dim=dim, keepdim=True)
    # For an all--inf row, use zero as the subtraction constant.  Its exponentials
    # are then all zero, and the guarded division below returns an all-zero row.
    subtraction = torch.where(torch.isfinite(maximum), maximum, torch.zeros_like(maximum))
    exponentials = torch.exp(working - subtraction)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    probabilities = exponentials / safe_denominator
    result = torch.where(denominator > 0, probabilities, torch.zeros_like(probabilities))
    return result.to(input_dtype) if input_dtype in (torch.float16, torch.bfloat16) else result


class RotaryPositionalEmbedding(nn.Module):
    """Apply adjacent-pair rotary positional embeddings to queries or keys."""

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

        self.theta = float(theta)
        self.d_k = int(d_k)
        self.max_seq_len = int(max_seq_len)

        dimension_indices = torch.arange(0, self.d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = self.theta ** (-dimension_indices / self.d_k)
        positions = torch.arange(self.max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]

        # These values are fixed and reproducible from the constructor arguments, so
        # keeping them out of checkpoints preserves the reference state_dict layout.
        self.register_buffer("inverse_frequencies", inverse_frequencies, persistent=False)
        self.register_buffer("cos_cache", torch.cos(angles), persistent=False)
        self.register_buffer("sin_cache", torch.sin(angles), persistent=False)

    def _angles_for_contiguous_positions(self, sequence_length: int) -> tuple[Tensor, Tensor]:
        """Return angles for positions ``0..sequence_length-1`` without a host sync."""

        if sequence_length <= self.max_seq_len:
            return self.cos_cache[:sequence_length], self.sin_cache[:sequence_length]

        positions = torch.arange(sequence_length, device=self.inverse_frequencies.device, dtype=torch.float32)
        angles = positions[:, None] * self.inverse_frequencies[None, :]
        return torch.cos(angles), torch.sin(angles)

    def _angles_for_positions(self, token_positions: Tensor) -> tuple[Tensor, Tensor]:
        positions = token_positions.to(device=self.cos_cache.device, dtype=torch.long)
        if positions.numel() == 0:
            shape = (*positions.shape, self.d_k // 2)
            empty = self.cos_cache.new_empty(shape)
            return empty, empty
        if torch.any(positions < 0):
            raise ValueError("token positions must be non-negative")

        # Use the precomputed cache in the common case, but permit callers to use
        # larger absolute positions (useful for chunked/incremental decoding).
        if int(positions.max().item()) < self.max_seq_len:
            return self.cos_cache[positions], self.sin_cache[positions]

        angles = positions.to(torch.float32).unsqueeze(-1) * self.inverse_frequencies
        return torch.cos(angles), torch.sin(angles)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_k:
            raise ValueError(f"expected shape (..., sequence_length, {self.d_k}), got {tuple(x.shape)}")
        if token_positions is None:
            cos_values, sin_values = self._angles_for_contiguous_positions(x.shape[-2])
        else:
            if token_positions.ndim == 0 or token_positions.shape[-1] != x.shape[-2]:
                raise ValueError("token_positions must have the same final (sequence) dimension as the input")
            if token_positions.ndim > x.ndim - 1:
                raise ValueError("token_positions has more batch dimensions than the input")

            # A query/key tensor often has an additional head batch dimension that is
            # absent from token_positions. Insert singleton axes immediately before the
            # sequence axis so that positions broadcast over all such dimensions.
            positions = token_positions
            while positions.ndim < x.ndim - 1:
                positions = positions.unsqueeze(-2)
            cos_values, sin_values = self._angles_for_positions(positions)

        cos_values = cos_values.to(device=x.device, dtype=torch.float32)
        sin_values = sin_values.to(device=x.device, dtype=torch.float32)

        input_dtype = x.dtype
        pairs = x.to(torch.float32).reshape(*x.shape[:-1], self.d_k // 2, 2)
        first, second = pairs[..., 0], pairs[..., 1]

        # Rotate each adjacent pair using the reference implementation's
        # row-vector convention.
        rotated_first = first * cos_values - second * sin_values
        rotated_second = first * sin_values + second * cos_values
        output = torch.stack((rotated_first, rotated_second), dim=-1).flatten(-2)
        return output.to(input_dtype)


# A concise alias is convenient in training/configuration code.
RoPE = RotaryPositionalEmbedding


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary batch-like dimensions."""

    if Q.ndim < 2 or K.ndim < 2 or V.ndim < 2:
        raise ValueError("Q, K, and V must each have at least two dimensions")
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have matching key dimensions")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must contain the same number of keys")
    if Q.shape[-1] == 0:
        raise ValueError("the key dimension must be non-zero")

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(Q.shape[-1])
    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("attention mask must have boolean dtype")
        mask = mask.to(device=scores.device)
        scores = scores.masked_fill(~mask, -torch.inf)

    attention_weights = softmax(scores, dim=-1)
    return torch.matmul(attention_weights, V)


class MultiHeadSelfAttention(nn.Module):
    """Bias-free causal multi-head self-attention, optionally with RoPE."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        max_seq_len: int | None = None,
        theta: float = 10_000.0,
        use_rope: bool | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0:
            raise ValueError("d_model and num_heads must both be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be evenly divisible by num_heads")

        self.d_model = int(d_model)
        self.num_heads = int(num_heads)
        self.d_head = self.d_model // self.num_heads
        if (
            self.d_head % 2 != 0
            and use_rope is not False
            and (rope is not None or use_rope is True or max_seq_len is not None)
        ):
            raise ValueError("the per-head dimension must be even when using RoPE")

        if use_rope is False:
            rope = None
        elif rope is None and max_seq_len is not None:
            rope = RotaryPositionalEmbedding(theta, self.d_head, max_seq_len, device=device)
        elif use_rope is True and rope is None:
            raise ValueError("use_rope=True requires either rope or max_seq_len")
        self.rope = rope

        mask_cache_length = max_seq_len
        if mask_cache_length is None and rope is not None:
            mask_cache_length = rope.max_seq_len
        if mask_cache_length is None:
            causal_mask = torch.empty((0, 0), device=device, dtype=torch.bool)
        else:
            mask_device = rope.cos_cache.device if rope is not None else device
            causal_mask = torch.ones(
                (mask_cache_length, mask_cache_length),
                device=mask_device,
                dtype=torch.bool,
            ).tril()
        self.register_buffer("_causal_mask", causal_mask, persistent=False)

        self.q_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.k_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.v_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)
        self.output_proj = Linear(self.d_model, self.d_model, device=device, dtype=dtype)

    def _split_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        split = x.reshape(*x.shape[:-2], sequence_length, self.num_heads, self.d_head)
        return split.transpose(-3, -2)

    def _merge_heads(self, x: Tensor) -> Tensor:
        sequence_length = x.shape[-2]
        merged = x.transpose(-3, -2)
        return merged.reshape(*merged.shape[:-3], sequence_length, self.d_model)

    def _get_causal_mask(self, sequence_length: int, device: torch.device) -> Tensor:
        cached = self._causal_mask
        if cached.device == device and cached.shape[0] >= sequence_length:
            return cached[:sequence_length, :sequence_length]

        new_length = max(sequence_length, cached.shape[0])
        self._causal_mask = torch.ones((new_length, new_length), device=device, dtype=torch.bool).tril()
        return self._causal_mask[:sequence_length, :sequence_length]

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if x.ndim < 2 or x.shape[-1] != self.d_model:
            raise ValueError(f"expected shape (..., sequence_length, {self.d_model}), got {tuple(x.shape)}")

        sequence_length = x.shape[-2]
        queries = self._split_heads(self.q_proj(x))
        keys = self._split_heads(self.k_proj(x))
        values = self._split_heads(self.v_proj(x))

        if self.rope is not None:
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        causal_mask = self._get_causal_mask(sequence_length, x.device)
        attended = scaled_dot_product_attention(queries, keys, values, causal_mask)
        return self.output_proj(self._merge_heads(attended))


CausalMultiHeadSelfAttention = MultiHeadSelfAttention


class TransformerBlock(nn.Module):
    """A pre-norm Transformer block with causal attention and SwiGLU."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int = 2048,
        theta: float = 10_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        rope: RotaryPositionalEmbedding | None = None,
        use_rope: bool = True,
        eps: float = 1e-5,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        ffn_type: str | None = None,
    ) -> None:
        super().__init__()
        if use_rope and rope is None:
            rope = RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)

        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
            max_seq_len=max_seq_len,
            use_rope=use_rope,
        )
        self.remove_rmsnorm = bool(remove_rmsnorm)
        self.use_post_norm = bool(use_post_norm)
        if self.remove_rmsnorm:
            self.ln1 = Identity()
            self.ln2 = Identity()
        else:
            self.ln1 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)

        normalized_ffn_type = "swiglu" if ffn_type is None else ffn_type.lower()
        if normalized_ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        elif normalized_ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        else:
            raise ValueError("ffn_type must be either 'swiglu', 'silu', or None")
        self.ffn_type = normalized_ffn_type

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.remove_rmsnorm:
            residual = x + self.attn(x, token_positions=token_positions)
            return residual + self.ffn(residual)

        if self.use_post_norm:
            residual = self.ln1(x + self.attn(x, token_positions=token_positions))
            return self.ln2(residual + self.ffn(residual))

        residual = x + self.attn(self.ln1(x), token_positions=token_positions)
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
        rope_theta: float = 10_000.0,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        use_rope: bool | None = None,
        eps: float = 1e-5,
        ffn_type: str | None = None,
    ) -> None:
        super().__init__()
        if vocab_size <= 0 or context_length <= 0 or num_layers < 0:
            raise ValueError("vocab_size and context_length must be positive, and num_layers non-negative")
        if d_model <= 0 or num_heads <= 0 or d_ff <= 0:
            raise ValueError("d_model, num_heads, and d_ff must all be positive")
        if d_model % num_heads != 0:
            raise ValueError("d_model must be evenly divisible by num_heads")

        if use_rope is not None:
            remove_rope = not use_rope
        self.vocab_size = int(vocab_size)
        self.context_length = int(context_length)
        self.d_model = int(d_model)
        self.num_layers = int(num_layers)

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        shared_rope = None
        if not remove_rope:
            shared_rope = RotaryPositionalEmbedding(
                rope_theta,
                d_model // num_heads,
                context_length,
                device=device,
            )

        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    rope=shared_rope,
                    use_rope=not remove_rope,
                    eps=eps,
                    remove_rmsnorm=remove_rmsnorm,
                    use_post_norm=use_post_norm,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        if remove_rmsnorm:
            self.ln_final = Identity()
        else:
            self.ln_final = RMSNorm(d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)
        self.remove_rmsnorm = bool(remove_rmsnorm)

    def forward(self, in_indices: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if in_indices.ndim < 1:
            raise ValueError("in_indices must have at least a sequence dimension")
        sequence_length = in_indices.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"input sequence length {sequence_length} exceeds context length {self.context_length}")
        if token_positions is not None and token_positions.shape[-1] != sequence_length:
            raise ValueError("token_positions and in_indices must have the same sequence length")

        hidden = self.token_embeddings(in_indices)
        for layer in self.layers:
            hidden = layer(hidden, token_positions=token_positions)
        if not self.remove_rmsnorm:
            hidden = self.ln_final(hidden)
        return self.lm_head(hidden)


TransformerLanguageModel = TransformerLM


__all__ = [
    "CausalMultiHeadSelfAttention",
    "Embedding",
    "Identity",
    "Linear",
    "MultiHeadSelfAttention",
    "RMSNorm",
    "RoPE",
    "RotaryPositionalEmbedding",
    "SiLU",
    "SiLUFeedForward",
    "SwiGLU",
    "TransformerBlock",
    "TransformerLM",
    "TransformerLanguageModel",
    "scaled_dot_product_attention",
    "silu",
    "softmax",
]
