from __future__ import annotations

import torch
from torch import Tensor, nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.modules import Embedding, Linear, RMSNorm, SiLUFeedForward, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        max_seq_len: int,
        theta: float,
        *,
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.remove_rmsnorm = remove_rmsnorm
        self.use_post_norm = use_post_norm
        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            max_seq_len=max_seq_len,
            theta=theta,
            use_rope=not remove_rope,
            device=device,
            dtype=dtype,
        )
        self.ffn = (
            SiLUFeedForward(d_model, d_ff, device=device, dtype=dtype)
            if ffn_type == "silu"
            else SwiGLU(d_model, d_ff, device=device, dtype=dtype)
        )
        self.ln1 = None if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.ln2 = None if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor, token_positions: Tensor | None = None) -> Tensor:
        if self.remove_rmsnorm:
            x = x + self.attn(x, token_positions)
            return x + self.ffn(x)
        assert self.ln1 is not None and self.ln2 is not None
        if self.use_post_norm:
            x = self.ln1(x + self.attn(x, token_positions))
            return self.ln2(x + self.ffn(x))
        x = x + self.attn(self.ln1(x), token_positions)
        return x + self.ffn(self.ln2(x))


class TransformerLM(nn.Module):
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
        remove_rmsnorm: bool = False,
        use_post_norm: bool = False,
        remove_rope: bool = False,
        ffn_type: str | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.context_length = context_length
        self.remove_rmsnorm = remove_rmsnorm
        self.token_embeddings = Embedding(vocab_size, d_model, device=device, dtype=dtype)
        self.layers = nn.ModuleList(
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
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = None if remove_rmsnorm else RMSNorm(d_model, device=device, dtype=dtype)
        self.lm_head = Linear(d_model, vocab_size, device=device, dtype=dtype)

    def forward(self, token_ids: Tensor) -> Tensor:
        sequence_length = token_ids.shape[-1]
        if sequence_length > self.context_length:
            raise ValueError("input sequence exceeds context_length")
        positions = torch.arange(sequence_length, device=token_ids.device)
        x = self.token_embeddings(token_ids)
        for layer in self.layers:
            x = layer(x, positions)
        if not self.remove_rmsnorm:
            assert self.ln_final is not None
            x = self.ln_final(x)
        return self.lm_head(x)
