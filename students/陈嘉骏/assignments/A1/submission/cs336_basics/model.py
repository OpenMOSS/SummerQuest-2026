from __future__ import annotations

import math
from typing import Literal

import torch
from torch import Tensor
from torch.nn import Module, ModuleList, Parameter


def _truncated_normal_(tensor: Tensor, std: float) -> None:
    """Initialize a tensor from N(0, std^2), truncated to three standard deviations."""
    lower_cdf = 0.5 * (1.0 + math.erf(-3.0 / math.sqrt(2.0)))
    upper_cdf = 0.5 * (1.0 + math.erf(3.0 / math.sqrt(2.0)))
    with torch.no_grad():
        values = torch.empty(tensor.shape, device=tensor.device, dtype=torch.float32)
        values.uniform_(2.0 * lower_cdf - 1.0, 2.0 * upper_cdf - 1.0)
        values.erfinv_().mul_(std * math.sqrt(2.0)).clamp_(-3.0 * std, 3.0 * std)
        tensor.copy_(values.to(dtype=tensor.dtype))


class Linear(Module):
    """Bias-free linear transformation with weight shape (d_out, d_in)."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("Linear dimensions must be positive.")
        self.in_features = in_features
        self.out_features = out_features
        self.d_in = in_features
        self.d_out = out_features
        self.weight = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        _truncated_normal_(self.weight, std=math.sqrt(2.0 / (in_features + out_features)))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_in:
            raise ValueError(f"Expected input dimension {self.d_in}, received {inputs.shape[-1]}.")
        return torch.matmul(inputs, self.weight.transpose(-1, -2))


class Embedding(Module):
    """Trainable lookup table implemented through tensor indexing."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("Embedding dimensions must be positive.")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _truncated_normal_(self.weight, std=1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive.")
        if eps <= 0:
            raise ValueError("eps must be positive.")
        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"Expected input dimension {self.d_model}, received {inputs.shape[-1]}.")
        input_dtype = inputs.dtype
        inputs_float = inputs.to(torch.float32)
        root_mean_square = torch.sqrt(torch.mean(inputs_float * inputs_float, dim=-1, keepdim=True) + self.eps)
        normalized = inputs_float / root_mean_square
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)


class Identity(Module):
    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


def silu(inputs: Tensor) -> Tensor:
    return inputs * torch.sigmoid(inputs)


class SwiGLU(Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_ff is None:
            d_ff = 64 * round((8.0 / 3.0 * d_model) / 64)
        if d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)) * self.w3(inputs))


class SiluFFN(Module):
    """Two-matrix SiLU feed-forward network used by the PDF ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        resolved_d_ff = 4 * d_model if d_ff is None else d_ff
        if resolved_d_ff <= 0:
            raise ValueError("d_ff must be positive.")
        self.d_model = d_model
        self.d_ff = resolved_d_ff
        self.w1 = Linear(d_model, resolved_d_ff, device=device, dtype=dtype)
        self.w2 = Linear(resolved_d_ff, d_model, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)))


def softmax(inputs: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax implemented from tensor primitives."""
    shifted = inputs - torch.amax(inputs, dim=dim, keepdim=True)
    exponentials = torch.exp(shifted)
    return exponentials / torch.sum(exponentials, dim=dim, keepdim=True)


def scaled_dot_product_attention(
    queries: Tensor,
    keys: Tensor,
    values: Tensor,
    mask: Tensor | None = None,
) -> Tensor:
    if queries.shape[-1] != keys.shape[-1]:
        raise ValueError("Query and key dimensions must match.")
    if keys.shape[-2] != values.shape[-2]:
        raise ValueError("Key and value sequence lengths must match.")

    scores = torch.matmul(queries, keys.transpose(-1, -2)) / math.sqrt(queries.shape[-1])
    if mask is None:
        attention_weights = softmax(scores, dim=-1)
    else:
        if mask.dtype != torch.bool:
            raise TypeError("Attention mask must have boolean dtype.")
        masked_scores = torch.where(mask, scores, torch.tensor(float("-inf"), device=scores.device, dtype=scores.dtype))
        row_maximum = torch.amax(masked_scores, dim=-1, keepdim=True)
        finite_row_maximum = torch.where(torch.isfinite(row_maximum), row_maximum, torch.zeros_like(row_maximum))
        exponentials = torch.where(mask, torch.exp(scores - finite_row_maximum), torch.zeros_like(scores))
        denominator = torch.sum(exponentials, dim=-1, keepdim=True)
        attention_weights = torch.where(
            denominator > 0,
            exponentials / torch.clamp(denominator, min=torch.finfo(scores.dtype).tiny),
            torch.zeros_like(exponentials),
        )
    return torch.matmul(attention_weights, values)


class RotaryPositionalEmbedding(Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device: torch.device | str | None = None,
    ) -> None:
        super().__init__()
        if d_k <= 0 or d_k % 2 != 0:
            raise ValueError("RoPE dimension must be a positive even integer.")
        if theta <= 0 or max_seq_len <= 0:
            raise ValueError("theta and max_seq_len must be positive.")
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        dimension_pairs = torch.arange(0, d_k, 2, device=device, dtype=torch.float32)
        inverse_frequencies = theta ** (-dimension_pairs / d_k)
        positions = torch.arange(max_seq_len, device=device, dtype=torch.float32)
        angles = positions[:, None] * inverse_frequencies[None, :]
        self.cosine: Tensor
        self.sine: Tensor
        self.register_buffer("cosine", torch.cos(angles), persistent=False)
        self.register_buffer("sine", torch.sin(angles), persistent=False)

    def forward(self, inputs: Tensor, token_positions: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_k:
            raise ValueError(f"Expected RoPE input dimension {self.d_k}, received {inputs.shape[-1]}.")
        if token_positions.shape[-1] != inputs.shape[-2]:
            raise ValueError("The final token_positions dimension must equal sequence length.")
        if token_positions.numel() and int(token_positions.max()) >= self.max_seq_len:
            raise ValueError("token_positions contains a position beyond max_seq_len.")

        cosine = self.cosine[token_positions].to(dtype=inputs.dtype)
        sine = self.sine[token_positions].to(dtype=inputs.dtype)
        while cosine.ndim < inputs.ndim:
            cosine = cosine.unsqueeze(-3)
            sine = sine.unsqueeze(-3)

        even = inputs[..., 0::2]
        odd = inputs[..., 1::2]
        output = torch.empty_like(inputs)
        output[..., 0::2] = even * cosine - odd * sine
        output[..., 1::2] = even * sine + odd * cosine
        return output


class MultiHeadSelfAttention(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        rope: RotaryPositionalEmbedding | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0 or num_heads <= 0 or d_model % num_heads != 0:
            raise ValueError("d_model must be positive and divisible by num_heads.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.q_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.k_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.v_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.output_proj = Linear(d_model, d_model, device=device, dtype=dtype)
        self.rope = rope

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = inputs.shape[-2]
        leading_shape = inputs.shape[:-2]

        queries = self._split_heads(self.q_proj(inputs))
        keys = self._split_heads(self.k_proj(inputs))
        values = self._split_heads(self.v_proj(inputs))

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(sequence_length, device=inputs.device)
            queries = self.rope(queries, token_positions)
            keys = self.rope(keys, token_positions)

        causal_mask = torch.ones(
            sequence_length,
            sequence_length,
            dtype=torch.bool,
            device=inputs.device,
        ).tril()
        attended = scaled_dot_product_attention(queries, keys, values, mask=causal_mask)
        attended = attended.transpose(-3, -2).contiguous()
        attended = attended.reshape(*leading_shape, sequence_length, self.d_model)
        return self.output_proj(attended)

    def _split_heads(self, inputs: Tensor) -> Tensor:
        sequence_length = inputs.shape[-2]
        leading_shape = inputs.shape[:-2]
        split = inputs.reshape(*leading_shape, sequence_length, self.num_heads, self.head_dim)
        return split.transpose(-3, -2)


class TransformerBlock(Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: Literal["swiglu", "silu"] | None = None,
    ) -> None:
        super().__init__()
        if ffn_type not in (None, "swiglu", "silu"):
            raise ValueError("ffn_type must be None, 'swiglu', or 'silu'.")
        rope = (
            None if remove_rope else RotaryPositionalEmbedding(theta, d_model // num_heads, max_seq_len, device=device)
        )
        self.attn = MultiHeadSelfAttention(d_model, num_heads, rope=rope, device=device, dtype=dtype)
        self.ffn = (
            SiluFFN(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "silu"
            else SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        )
        self.ln1 = Identity() if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = Identity() if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.use_post_norm = use_post_norm

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.use_post_norm:
            attended = self.ln1(inputs + self.attn(inputs, token_positions=token_positions))
            return self.ln2(attended + self.ffn(attended))
        attended = inputs + self.attn(self.ln1(inputs), token_positions=token_positions)
        return attended + self.ffn(self.ln2(attended))


class TransformerLM(Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: Literal["swiglu", "silu"] | None = None,
    ) -> None:
        super().__init__()
        if context_length <= 0 or num_layers <= 0:
            raise ValueError("context_length and num_layers must be positive.")
        self.context_length = context_length
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    device=device,
                    dtype=dtype,
                    remove_rmsnorm=remove_rmsnorm,
                    use_post_norm=use_post_norm,
                    remove_rope=remove_rope,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = Identity() if remove_rmsnorm or use_post_norm else RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"Input sequence length {sequence_length} exceeds context length {self.context_length}.")
        token_positions = torch.arange(sequence_length, device=token_ids.device)
        hidden_states = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden_states = layer(hidden_states, token_positions=token_positions)
        return self.lm_head(self.ln_final(hidden_states))
