import torch
import torch.nn as nn

from .embedding import Embedding
from .linear import Linear
from .rms_norm import RMSNorm
from .transformer_block import TransformerBlock


class TransformerLM(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        vocab_size: int,
        num_layers: int,
        max_seq_len: int | None = None,
        d_ff: int | None = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        eps: float | None = None,
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
        self.context_length = max_seq_len
        self.norm_mode = norm_mode
        self.ffn_type = ffn_type

        # 1. embedding
        self.token_embeddings = Embedding(num_embeddings=vocab_size, embedding_dim=d_model, device=device, dtype=dtype)

        # 2. num_layers * transformer_block
        self.layers = nn.ModuleList(
            [
                TransformerBlock(
                    d_model=d_model,
                    num_heads=num_heads,
                    d_ff=d_ff,
                    max_seq_len=max_seq_len,
                    theta=theta,
                    use_rope=use_rope,
                    eps=self.eps,
                    device=device,
                    dtype=dtype,
                    norm_mode=norm_mode,
                    ffn_type=ffn_type,
                )
                for _ in range(num_layers)
            ]
        )

        # 3. RmsNorm
        self.ln_final = (
            None if norm_mode == "none" else RMSNorm(eps=self.eps, d_model=d_model, device=device, dtype=dtype)
        )
        # 4. output_linear
        self.lm_head = Linear(in_features=d_model, out_features=vocab_size, device=device, dtype=dtype)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert x.dtype == torch.long
        # 1. token embedding
        y = self.token_embeddings(x)
        # 2. Transformer_Block
        for block in self.layers:
            y = block(x=y, token_positions=token_positions)
        # 3. rmsNorm
        if self.ln_final is not None:
            y = self.ln_final(y)
        # 4.output embedding
        output = self.lm_head(y)
        # 5. softmax
        return output
