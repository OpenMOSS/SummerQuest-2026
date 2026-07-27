from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from einops import rearrange
from torch import Tensor, nn


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax along a selected tensor dimension."""
    input_dtype = x.dtype
    computation = (
        x.to(torch.float32)
        if x.dtype in (torch.float16, torch.bfloat16)
        else x
    )
    maximum = computation.amax(dim=dim, keepdim=True)
    exponentials = torch.exp(computation - maximum)
    probabilities = exponentials / exponentials.sum(dim=dim, keepdim=True)
    return probabilities.to(input_dtype)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Average cross-entropy over all batch-like dimensions."""
    if inputs.ndim < 1:
        raise ValueError("inputs must have a vocabulary dimension.")
    if inputs.shape[:-1] != targets.shape:
        raise ValueError(
            "targets must have the same shape as inputs without the final "
            "vocabulary dimension."
        )
    if inputs.shape[-1] == 0:
        raise ValueError("The vocabulary dimension must be positive.")

    input_dtype = inputs.dtype
    computation = (
        inputs.to(torch.float32)
        if inputs.dtype in (torch.float16, torch.bfloat16)
        else inputs
    )

    maximum = computation.amax(dim=-1, keepdim=True)
    shifted = computation - maximum
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    target_logits = shifted.gather(
        dim=-1,
        index=targets.to(dtype=torch.long).unsqueeze(-1),
    ).squeeze(-1)

    loss = (log_normalizer - target_logits).mean()
    return loss.to(input_dtype)


def scaled_dot_product_attention(
    Q: Tensor,
    K: Tensor,
    V: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    """Compute scaled dot-product attention over arbitrary batch-like dimensions."""
    if Q.shape[-1] != K.shape[-1]:
        raise ValueError("Q and K must have the same final dimension.")
    if K.shape[-2] != V.shape[-2]:
        raise ValueError("K and V must have the same number of key positions.")

    d_k = Q.shape[-1]
    if d_k == 0:
        raise ValueError("The query/key feature dimension must be positive.")

    scores = torch.einsum("...qd,...kd->...qk", Q, K)
    scores = scores / math.sqrt(d_k)

    if mask is not None:
        if mask.dtype != torch.bool:
            raise TypeError("Attention mask must have boolean dtype.")
        scores = scores.masked_fill(~mask, float("-inf"))

    attention_weights = softmax(scores, dim=-1)
    if mask is not None:
        attention_weights = torch.where(
            mask,
            attention_weights,
            torch.zeros((), dtype=attention_weights.dtype, device=attention_weights.device),
        )

    return torch.einsum("...qk,...kv->...qv", attention_weights, V)


class Linear(nn.Module):
    """Bias-free linear transformation with batch-like leading dimensions."""

    def __init__(self, in_features: int, out_features: int, device=None, dtype=None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(
            torch.empty(
                out_features,
                in_features,
                device=device,
                dtype=dtype,
            )
        )

        std = (2.0 / (in_features + out_features)) ** 0.5
        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=std,
            a=-3.0 * std,
            b=3.0 * std,
        )

    def forward(self, x: Tensor) -> Tensor:
        return torch.einsum(
            "...i,oi->...o",
            x,
            self.weight,
        )


class Embedding(nn.Module):
    """Embedding lookup from integer token IDs to dense vectors."""

    def __init__(self, num_embeddings: int, embedding_dim: int, device=None, dtype=None):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim

        self.weight = nn.Parameter(
            torch.empty(
                num_embeddings,
                embedding_dim,
                device=device,
                dtype=dtype,
            )
        )

        nn.init.trunc_normal_(
            self.weight,
            mean=0.0,
            std=1.0,
            a=-3.0,
            b=3.0,
        )

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root Mean Square normalization over the final feature dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(
            torch.ones(
                d_model,
                device=device,
                dtype=dtype,
            )
        )

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(
            x_float.pow(2).mean(dim=-1, keepdim=True) + self.eps
        )
        result = x_float / rms * self.weight.to(torch.float32)
        return result.to(input_dtype)


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network without bias terms."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        activated = F.silu(self.w1(x))
        gated = self.w3(x)
        return self.w2(activated * gated)


class SiLUFeedForward(nn.Module):
    """Position-wise two-layer SiLU feed-forward network without gating."""

    def __init__(self, d_model: int, d_ff: int, device=None, dtype=None):
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(F.silu(self.w1(x)))


# Descriptive alias used by some callers for the same position-wise module.
PositionWiseFeedForward = SwiGLU


class RotaryPositionalEmbedding(nn.Module):
    """Apply rotary positional rotations to the final dimension of a tensor."""

    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()
        if d_k % 2 != 0:
            raise ValueError("RoPE requires an even embedding dimension.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be positive.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        pair_indices = torch.arange(
            0,
            d_k,
            2,
            device=device,
            dtype=torch.float32,
        )
        inverse_frequencies = theta ** (-pair_indices / d_k)
        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )
        angles = positions[:, None] * inverse_frequencies[None, :]

        self.register_buffer("cos_cached", angles.cos(), persistent=False)
        self.register_buffer("sin_cached", angles.sin(), persistent=False)

    def forward(self, x: Tensor, token_positions: Tensor) -> Tensor:
        if x.shape[-1] != self.d_k:
            raise ValueError(
                f"Expected final dimension {self.d_k}, got {x.shape[-1]}"
            )
        if token_positions.ndim == 0 or token_positions.shape[-1] != x.shape[-2]:
            raise ValueError(
                "token_positions must have a final dimension matching x's sequence length"
            )
        if token_positions.numel() and (
            token_positions.min() < 0
            or token_positions.max() >= self.max_seq_len
        ):
            raise ValueError("token_positions contains an out-of-range position.")

        cos = self.cos_cached[token_positions].to(dtype=x.dtype)
        sin = self.sin_cached[token_positions].to(dtype=x.dtype)

        even = x[..., 0::2]
        odd = x[..., 1::2]
        rotated_even = even * cos - odd * sin
        rotated_odd = even * sin + odd * cos

        output = torch.empty_like(x)
        output[..., 0::2] = rotated_even
        output[..., 1::2] = rotated_odd
        return output


class CausalMultiHeadSelfAttention(nn.Module):
    """Causal multi-head self-attention with optional rotary position encoding."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
        use_rope: bool = True,
    ):
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_head = d_model // num_heads
        self.use_rope = use_rope

        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)

        self.rope = (
            RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.d_head,
                max_seq_len=max_seq_len,
                device=device,
            )
            if use_rope
            else None
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = x.shape[-2]
        q = rearrange(
            self.q_proj(x),
            "... sequence (heads d_head) -> ... heads sequence d_head",
            heads=self.num_heads,
            d_head=self.d_head,
        )
        k = rearrange(
            self.k_proj(x),
            "... sequence (heads d_head) -> ... heads sequence d_head",
            heads=self.num_heads,
            d_head=self.d_head,
        )
        v = rearrange(
            self.v_proj(x),
            "... sequence (heads d_head) -> ... heads sequence d_head",
            heads=self.num_heads,
            d_head=self.d_head,
        )

        if self.use_rope:
            if token_positions is None:
                token_positions = torch.arange(
                    sequence_length,
                    device=x.device,
                )
            assert self.rope is not None
            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        positions = torch.arange(sequence_length, device=x.device)
        causal_mask = positions[None, :] <= positions[:, None]
        attended = scaled_dot_product_attention(q, k, v, mask=causal_mask)
        merged = rearrange(
            attended,
            "... heads sequence d_head -> ... sequence (heads d_head)",
        )
        return self.output_proj(merged)


class TransformerBlock(nn.Module):
    """A Transformer block with configurable RMSNorm placement."""

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float = 10000.0,
        max_seq_len: int = 2048,
        device=None,
        dtype=None,
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be 'swiglu' or 'silu'.")
        self.use_rmsnorm = use_rmsnorm
        self.pre_norm = pre_norm
        self.use_rope = use_rope
        self.ffn_type = ffn_type
        self.ln1 = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else None
        )
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
            use_rope=use_rope,
        )
        self.ln2 = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else None
        )
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.ln1 is None or self.ln2 is None:
            # With RMSNorm removed, pre/post placement has no effect.
            x = x + self.attn(x, token_positions=token_positions)
            x = x + self.ffn(x)
        elif self.pre_norm:
            x = x + self.attn(self.ln1(x), token_positions=token_positions)
            x = x + self.ffn(self.ln2(x))
        else:
            x = self.ln1(
                x + self.attn(x, token_positions=token_positions)
            )
            x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    """Decoder-only Transformer language model producing next-token logits."""

    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device=None,
        dtype=None,
        use_rmsnorm: bool = True,
        pre_norm: bool = True,
        use_rope: bool = True,
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive.")
        if context_length <= 0:
            raise ValueError("context_length must be positive.")
        if num_layers <= 0:
            raise ValueError("num_layers must be positive.")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.use_rmsnorm = use_rmsnorm
        self.pre_norm = pre_norm
        self.use_rope = use_rope
        self.ffn_type = ffn_type

        self.token_embeddings = Embedding(
            num_embeddings=vocab_size,
            embedding_dim=d_model,
            device=device,
            dtype=dtype,
        )
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    theta=rope_theta,
                    max_seq_len=context_length,
                    device=device,
                    dtype=dtype,
                    use_rmsnorm=use_rmsnorm,
                    pre_norm=pre_norm,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = (
            RMSNorm(d_model, device=device, dtype=dtype)
            if use_rmsnorm
            else None
        )
        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, in_indices: Tensor) -> Tensor:
        if in_indices.ndim < 1:
            raise ValueError("in_indices must have a sequence dimension.")

        sequence_length = in_indices.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(
                f"Input sequence length {sequence_length} exceeds context length "
                f"{self.context_length}."
            )

        token_positions = torch.arange(
            sequence_length,
            device=in_indices.device,
        )
        x = self.token_embeddings(in_indices)
        for layer in self.layers:
            x = layer(x, token_positions=token_positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
