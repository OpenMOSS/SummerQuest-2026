from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from cs336_basics.Part3.multihead_self_attention import MultiHeadSelfAttention
from cs336_basics.Part3.multihead_self_attention_with_rope import MultiHeadSelfAttentionWithRoPE
from cs336_basics.Part3.positionwise_feedforward import SiLUFeedForward, SwiGLU
from cs336_basics.Part3.rmsnorm import RMSNorm


NormMode = Literal["pre", "post", "none"]
FFNType = Literal["swiglu", "silu"]


class TransformerBlock(nn.Module):
    """Configurable causal Transformer block used by the ablation runs."""

    d_model: int
    num_heads: int
    d_ff: int
    norm_mode: NormMode
    use_rope: bool
    ffn_type: FFNType
    ffn_hidden_dim: int
    attn: MultiHeadSelfAttentionWithRoPE | MultiHeadSelfAttention
    ffn: SwiGLU | SiLUFeedForward
    ln1: RMSNorm | None
    ln2: RMSNorm | None

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float | None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        *,
        norm_mode: NormMode = "pre",
        use_rope: bool = True,
        ffn_type: FFNType = "swiglu",
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if d_ff <= 0:
            raise ValueError("d_ff must be greater than zero.")
        if max_seq_len <= 0:
            raise ValueError("max_seq_len must be greater than zero.")
        if norm_mode not in ("pre", "post", "none"):
            raise ValueError("norm_mode must be one of: 'pre', 'post', 'none'.")
        if ffn_type not in ("swiglu", "silu"):
            raise ValueError("ffn_type must be one of: 'swiglu', 'silu'.")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.norm_mode = norm_mode
        self.use_rope = use_rope
        self.ffn_type = ffn_type

        if use_rope:
            if theta is None or theta <= 0:
                raise ValueError("theta must be greater than zero when use_rope is True.")
            self.attn = MultiHeadSelfAttentionWithRoPE(
                d_model,
                num_heads,
                max_seq_len,
                theta,
                device=device,
                dtype=dtype,
            )
        else:
            self.attn = MultiHeadSelfAttention(d_model, num_heads, device=device, dtype=dtype)

        if ffn_type == "swiglu":
            self.ffn_hidden_dim = d_ff
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            self.ffn_hidden_dim = SiLUFeedForward.matched_hidden_dim(d_ff)
            self.ffn = SiLUFeedForward(d_model, self.ffn_hidden_dim, device=device, dtype=dtype)

        if norm_mode == "none":
            self.ln1 = None
            self.ln2 = None
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def _run_attention(self, x: torch.Tensor, token_positions: torch.Tensor | None) -> torch.Tensor:
        if self.use_rope:
            return self.attn(x, token_positions)
        return self.attn(x)

    def forward(self, x: torch.Tensor, token_positions: torch.Tensor | None = None) -> torch.Tensor:
        """Transform an input of shape ``(..., sequence_length, d_model)``."""

        if x.ndim < 2:
            raise ValueError("x must include sequence and feature dimensions.")
        if x.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {x.shape[-1]}.")

        if self.norm_mode == "none":
            x = x + self._run_attention(x, token_positions)
            return x + self.ffn(x)

        if self.ln1 is None or self.ln2 is None:
            raise RuntimeError("normalization modules are missing for the configured norm mode.")

        if self.norm_mode == "post":
            x = self.ln1(x + self._run_attention(x, token_positions))
            return self.ln2(x + self.ffn(x))

        x = x + self._run_attention(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))
