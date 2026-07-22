from __future__ import annotations

import torch
from torch import Tensor, nn

from .attention import CausalMultiHeadSelfAttention
from .nn import Embedding, Linear, RMSNorm, SiLUFeedForward, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float = 10_000.0,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if norm_style not in {"pre", "post", "none"}:
            raise ValueError("norm_style must be one of: pre, post, none")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be one of: swiglu, silu")
        self.norm_style = norm_style
        self.attn = CausalMultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, theta, use_rope=use_rope, device=device, dtype=dtype
        )
        self.ln1 = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None
        self.ln2 = RMSNorm(d_model, device=device, dtype=dtype) if norm_style != "none" else None
        self.ffn = (
            SwiGLU(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "swiglu"
            else SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
        )

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.norm_style == "pre":
            assert self.ln1 is not None and self.ln2 is not None
            x = x + self.attn(self.ln1(x), token_positions)
            return x + self.ffn(self.ln2(x))
        if self.norm_style == "post":
            assert self.ln1 is not None and self.ln2 is not None
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(x, token_positions)
        return x + self.ffn(x)


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        rope_theta: float = 10_000.0,
        norm_style: str = "pre",
        use_rope: bool = True,
        ffn_type: str = "swiglu",
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.norm_style = norm_style
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model,
                    num_heads,
                    d_ff,
                    context_length,
                    rope_theta,
                    norm_style=norm_style,
                    use_rope=use_rope,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, device=device, dtype=dtype) if norm_style == "pre" else None
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor, token_positions: Tensor | None = None) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError("input sequence exceeds context_length")
        if token_positions is None:
            token_positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, token_positions)
        if self.ln_final is not None:
            x = self.ln_final(x)
        return self.lm_head(x)
