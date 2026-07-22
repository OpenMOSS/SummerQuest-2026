"""Pre/post/no-norm decoder-only Transformer language model."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor, nn

from .attention import MultiHeadSelfAttention, RotaryPositionalEmbedding
from .nn import Embedding, Linear, RMSNorm, SiLUFeedForward, SwiGLU

NormStyle = Literal["pre", "post", "none"]
PositionEncoding = Literal["rope", "none"]
FFNType = Literal["swiglu", "silu"]


def _validate_architecture_options(
    norm_style: str,
    position_encoding: str,
    ffn_type: str,
) -> None:
    if norm_style not in {"pre", "post", "none"}:
        raise ValueError(f"norm_style must be 'pre', 'post', or 'none', got {norm_style!r}")
    if position_encoding not in {"rope", "none"}:
        raise ValueError(f"position_encoding must be 'rope' or 'none', got {position_encoding!r}")
    if ffn_type not in {"swiglu", "silu"}:
        raise ValueError(f"ffn_type must be 'swiglu' or 'silu', got {ffn_type!r}")


class TransformerBlock(nn.Module):
    """A causal Transformer block with configurable architecture ablations.

    Defaults implement the assignment's pre-norm, RoPE, SwiGLU block.  The
    alternative settings are intended for the required architecture ablations.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        *,
        norm_style: NormStyle = "pre",
        position_encoding: PositionEncoding = "rope",
        ffn_type: FFNType = "swiglu",
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_architecture_options(norm_style, position_encoding, ffn_type)
        if d_model <= 0 or num_heads <= 0 or d_ff <= 0 or max_seq_len <= 0:
            raise ValueError("all Transformer dimensions must be positive")
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.theta = theta
        self.norm_style = norm_style
        self.position_encoding = position_encoding
        self.ffn_type = ffn_type

        rope = None
        if position_encoding == "rope":
            rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=d_model // num_heads,
                max_seq_len=max_seq_len,
                device=device,
            )
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            rope=rope,
            device=device,
            dtype=dtype,
        )
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFeedForward(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

        if norm_style == "none":
            self.ln1 = None
            self.ln2 = None
        else:
            self.ln1 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if inputs.shape[-2] > self.max_seq_len:
            raise ValueError(f"sequence length {inputs.shape[-2]} exceeds maximum context length {self.max_seq_len}")

        if self.norm_style == "pre":
            assert self.ln1 is not None and self.ln2 is not None
            hidden = inputs + self.attn(self.ln1(inputs), token_positions)
            return hidden + self.ffn(self.ln2(hidden))

        if self.norm_style == "post":
            assert self.ln1 is not None and self.ln2 is not None
            hidden = self.ln1(inputs + self.attn(inputs, token_positions))
            return self.ln2(hidden + self.ffn(hidden))

        hidden = inputs + self.attn(inputs, token_positions)
        return hidden + self.ffn(hidden)


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
        *,
        norm_style: NormStyle = "pre",
        position_encoding: PositionEncoding = "rope",
        ffn_type: FFNType = "swiglu",
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        _validate_architecture_options(norm_style, position_encoding, ffn_type)
        if vocab_size <= 0 or context_length <= 0 or d_model <= 0 or num_layers <= 0:
            raise ValueError("vocab_size, context_length, d_model, and num_layers must be positive")

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.norm_style = norm_style
        self.position_encoding = position_encoding
        self.ffn_type = ffn_type

        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    theta=rope_theta,
                    norm_style=norm_style,
                    position_encoding=position_encoding,
                    ffn_type=ffn_type,
                    eps=eps,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = None if norm_style == "none" else RMSNorm(d_model=d_model, eps=eps, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError(f"sequence length {sequence_length} exceeds context length {self.context_length}")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=token_ids.device)
        elif token_positions.shape[-1] != sequence_length:
            raise ValueError(
                "token_positions and token_ids must have the same sequence length, "
                f"got {token_positions.shape[-1]} and {sequence_length}"
            )

        hidden = self.token_embeddings(token_ids)
        for layer in self.layers:
            hidden = layer(hidden, token_positions)
        if self.ln_final is not None:
            hidden = self.ln_final(hidden)
        return self.lm_head(hidden)


__all__ = [
    "FFNType",
    "NormStyle",
    "PositionEncoding",
    "TransformerBlock",
    "TransformerLM",
]
