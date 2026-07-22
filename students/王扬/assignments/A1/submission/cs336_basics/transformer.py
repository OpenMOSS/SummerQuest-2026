from __future__ import annotations

from torch import Tensor, nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.nn import Embedding, Linear, RMSNorm, SiLUFeedForward, SwiGLU


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        *,
        max_seq_len: int,
        rope_theta: float,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        prenorm: bool = True,
        ffn_type: str = "swiglu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.max_seq_len = max_seq_len
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.prenorm = prenorm
        self.ffn_type = ffn_type

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            use_rope=use_rope,
            rope_theta=rope_theta,
            max_seq_len=max_seq_len,
        )
        self.ln1 = RMSNorm(d_model, eps=eps) if use_rmsnorm else nn.Identity()
        if ffn_type == "swiglu":
            self.ffn = SwiGLU(d_model, d_ff)
        elif ffn_type == "silu":
            self.ffn = SiLUFeedForward(d_model, d_ff)
        else:
            raise ValueError("Unknown ffn_type")
        self.ln2 = RMSNorm(d_model, eps=eps) if use_rmsnorm else nn.Identity()

    def forward(
        self,
        x: Tensor,
        token_positions: Tensor | None = None,
        mask: Tensor | None = None,
    ) -> Tensor:
        if self.prenorm:
            x = x + self.attn(self.ln1(x), token_positions=token_positions, mask=mask)
            x = x + self.ffn(self.ln2(x))
            return x
        x = self.ln1(x + self.attn(x, token_positions=token_positions, mask=mask))
        x = self.ln2(x + self.ffn(x))
        return x


class TransformerLM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        context_length: int,
        d_model: int,
        num_layers: int,
        num_heads: int,
        d_ff: int,
        *,
        rope_theta: float,
        use_rope: bool = True,
        use_rmsnorm: bool = True,
        prenorm: bool = True,
        ffn_type: str = "swiglu",
        eps: float = 1e-5,
    ) -> None:
        super().__init__()
        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.use_rope = use_rope
        self.use_rmsnorm = use_rmsnorm
        self.prenorm = prenorm
        self.ffn_type = ffn_type

        self.token_embeddings = Embedding(vocab_size, d_model)
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=context_length,
                    rope_theta=rope_theta,
                    use_rope=use_rope,
                    use_rmsnorm=use_rmsnorm,
                    prenorm=prenorm,
                    ffn_type=ffn_type,
                    eps=eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.ln_final = RMSNorm(d_model, eps=eps) if use_rmsnorm else nn.Identity()
        self.lm_head = Linear(d_model, vocab_size)

    def forward(self, token_ids: Tensor) -> Tensor:
        seq_len = token_ids.shape[-1]
        if seq_len > self.context_length:
            raise ValueError("Input sequence length exceeds context length.")

        x = self.token_embeddings(token_ids)
        token_positions = token_ids.new_tensor(range(seq_len))
        token_positions = token_positions.view(*([1] * (token_ids.ndim - 1)), seq_len).expand_as(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions=token_positions)

        x = self.ln_final(x)
        return self.lm_head(x)
