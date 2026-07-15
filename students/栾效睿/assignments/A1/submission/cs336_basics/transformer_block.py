import torch
import torch.nn as nn

from .causal_multiHead_selfAttention import CausalMultiHeadSelfAttention
from .rms_norm import RMSNorm
from .silu_ffn import SiLUFFN
from .swiglu import SwiGlu


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        d_ff: int | None = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        eps: float | None = 1e-5,
        device=None,
        dtype=None,
        norm_mode: str = "pre",
        ffn_type: str = "swiglu",
    ):
        super().__init__()
        if norm_mode not in {"pre", "post", "none"}:
            raise ValueError(f"norm_mode must be one of 'pre', 'post', or 'none', got {norm_mode!r}")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError(f"ffn_type must be either 'swiglu' or 'silu', got {ffn_type!r}")

        self.eps = eps if eps is not None else 1e-5
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type
        self.ln1 = None if norm_mode == "none" else RMSNorm(eps=self.eps, d_model=d_model, device=device, dtype=dtype)
        self.attn = CausalMultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=use_rope,
            device=device,
            dtype=dtype,
        )
        self.ln2 = None if norm_mode == "none" else RMSNorm(eps=self.eps, d_model=d_model, device=device, dtype=dtype)
        if ffn_type == "swiglu":
            self.ffn = SwiGlu(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)
        else:
            self.ffn = SiLUFFN(d_model=d_model, d_ff=d_ff, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_mode == "pre":
            assert self.ln1 is not None and self.ln2 is not None
            y = x + self.attn(x=self.ln1(x), token_positions=token_positions)
            return y + self.ffn(self.ln2(y))

        if self.norm_mode == "post":
            assert self.ln1 is not None and self.ln2 is not None
            y = self.ln1(x + self.attn(x=x, token_positions=token_positions))
            return self.ln2(y + self.ffn(y))

        y = x + self.attn(x=x, token_positions=token_positions)
        return y + self.ffn(y)
