import torch
import torch.nn as nn
from einops import rearrange
from .linear import Linear
from .rope import RoPE
from .scaled_dot_product_attention import scaled_dot_product_attention


class CausalMultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        max_seq_len: int | None = None,
        theta: float = 10000.0,
        use_rope: bool = True,
        device=None,
        dtype=None,
    ):
        super().__init__()
        assert d_model % num_heads == 0
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.q_proj = Linear(
            in_features=d_model, out_features=d_model, device=device, dtype=dtype
        )
        self.k_proj = Linear(
            in_features=d_model, out_features=d_model, device=device, dtype=dtype
        )
        self.v_proj = Linear(
            in_features=d_model, out_features=d_model, device=device, dtype=dtype
        )
        self.output_proj = Linear(
            in_features=d_model, out_features=d_model, device=device, dtype=dtype
        )
        self.use_rope = use_rope
        if self.use_rope:
            if self.d_k % 2 != 0:
                raise ValueError("d_model // num_heads must be even when using RoPE")
            self.rope = RoPE(
                theta=theta,
                d_k=self.d_k,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        # project Q, k, V
        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        # Split d_model into num_heads * d_k
        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        # RoPE
        if self.use_rope and self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(seq_len, device=x.device)

            q = self.rope(token_positions=token_positions, x=q)
            k = self.rope(token_positions=token_positions, x=k)

        # cal attention
        causal_mask = torch.tril(
            torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool)
        )
        output = scaled_dot_product_attention(masked=causal_mask, Q=q, K=k, V=v)
        # merge num_heads * d_k to d_model
        output = self._merge_heads(output)
        return self.output_proj(output)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        # input:  (..., seq_len, d_model)
        # output: (..., num_heads, seq_len, d_k)
        return rearrange(
            x,
            "... seq (h d) -> ... h seq d",
            h=self.num_heads,
        )

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "... h seq d -> ... seq (h d)")
