import torch
from torch import nn

from cs336_basics.attention import MultiHeadSelfAttention
from cs336_basics.nn import Embedding, Identity, Linear, RMSNorm, SiLUFFN, SwiGLU

class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        d_ff: int,
        theta: float,
        max_seq_len: int,
        eps: float = 1e-5,
        use_rope: bool = True,
        norm_position: str = "pre",
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()

        if norm_position not in {"pre", "post", "none"}:
            raise ValueError("norm_position must be one of: pre, post, none")
        if ffn_type not in {"swiglu", "silu"}:
            raise ValueError("ffn_type must be one of: swiglu, silu")

        self.norm_position = norm_position
        norm_cls = Identity if norm_position == "none" else RMSNorm

        self.ln1 = norm_cls(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        ) if norm_cls is RMSNorm else norm_cls()

        self.attn = MultiHeadSelfAttention(
            d_model=d_model,
            num_heads=num_heads,
            theta=theta if use_rope else None,
            max_seq_len=max_seq_len,
            device=device,
            dtype=dtype,
        )

        self.ln2 = norm_cls(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        ) if norm_cls is RMSNorm else norm_cls()

        ffn_cls = SwiGLU if ffn_type == "swiglu" else SiLUFFN
        self.ffn = ffn_cls(
            d_model=d_model,
            d_ff=d_ff,
            device=device,
            dtype=dtype,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if self.norm_position == "post":
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
        rope_theta: float = 10000.0,
        eps: float = 1e-5,
        use_rope: bool = True,
        norm_position: str = "pre",
        use_final_norm: bool = True,
        ffn_type: str = "swiglu",
        device=None,
        dtype=None,
    ):
        super().__init__()

        self.vocab_size = vocab_size
        self.context_length = context_length
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.d_ff = d_ff
        self.rope_theta = rope_theta
        self.use_rope = use_rope
        self.norm_position = norm_position
        self.use_final_norm = use_final_norm
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
                    eps=eps,
                    use_rope=use_rope,
                    norm_position=norm_position,
                    ffn_type=ffn_type,
                    device=device,
                    dtype=dtype,
                )
                for _ in range(num_layers)
            ]
        )

        self.ln_final = RMSNorm(
            d_model=d_model,
            eps=eps,
            device=device,
            dtype=dtype,
        ) if use_final_norm else Identity()

        self.lm_head = Linear(
            in_features=d_model,
            out_features=vocab_size,
            device=device,
            dtype=dtype,
        )

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        *leading_dims, seq_len = token_ids.shape

        token_positions = torch.arange(
            seq_len,
            device=token_ids.device,
            dtype=torch.long,
        )

        token_positions = token_positions.reshape(
            *([1] * len(leading_dims)),
            seq_len,
        )

        token_positions = token_positions.expand(
            *leading_dims,
            seq_len,
        )

        x = self.token_embeddings(token_ids)

        for layer in self.layers:
            x = layer(x, token_positions)

        x = self.ln_final(x)

        logits = self.lm_head(x)

        return logits
