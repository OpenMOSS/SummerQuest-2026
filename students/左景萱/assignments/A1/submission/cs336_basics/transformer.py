"""Transformer block used by the causal language model."""

from __future__ import annotations

from torch import Tensor, nn

from .attention import MultiHeadSelfAttention
from .nn import Identity, RMSNorm, SiLUFeedForward, SwiGLU


class TransformerBlock(nn.Module):
    """A causal Transformer block supporting the assignment ablations."""

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
        norm_mode: str | None = None,
        use_rope: bool | None = None,
        ffn_type: str | None = "swiglu",
        device=None,
        dtype=None,
    ) -> None:
        super().__init__()
        if norm_mode is not None:
            norm_mode = norm_mode.lower()
            if norm_mode not in {"pre", "post", "none"}:
                raise ValueError("norm_mode must be 'pre', 'post', or 'none'")
            remove_rmsnorm = norm_mode == "none"
            use_post_norm = norm_mode == "post"
        if use_rope is not None:
            remove_rope = not use_rope
        if remove_rmsnorm and use_post_norm:
            raise ValueError("post-norm cannot be enabled when RMSNorm is removed")
        normalized_ffn_type = "swiglu" if ffn_type is None else ffn_type.lower().replace("_", "-")
        if normalized_ffn_type not in {"swiglu", "silu", "matched-silu"}:
            raise ValueError("ffn_type must be 'swiglu', 'silu', or 'matched-silu'")

        self.remove_rmsnorm = remove_rmsnorm
        self.use_post_norm = use_post_norm
        self.attn = MultiHeadSelfAttention(
            d_model,
            num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=not remove_rope,
            device=device,
            dtype=dtype,
        )
        if normalized_ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        else:
            # SwiGLU has 3*d_model*d_ff parameters. A two-projection FFN
            # matches that count when its hidden width is 3*d_ff/2.
            silu_width = (3 * d_ff + 1) // 2 if normalized_ffn_type == "matched-silu" else d_ff
            self.ffn = SiLUFeedForward(d_model, silu_width, device=device, dtype=dtype)

        if remove_rmsnorm:
            self.ln1 = Identity()
            self.ln2 = Identity()
        else:
            self.ln1 = RMSNorm(d_model, device=device, dtype=dtype)
            self.ln2 = RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, inputs: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.use_post_norm:
            hidden = self.ln1(inputs + self.attn(inputs, token_positions))
            return self.ln2(hidden + self.ffn(hidden))

        hidden = inputs + self.attn(self.ln1(inputs), token_positions)
        return hidden + self.ffn(self.ln2(hidden))


__all__ = ["TransformerBlock"]
