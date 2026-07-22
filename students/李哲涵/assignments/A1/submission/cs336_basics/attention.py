import math
import torch
from torch import nn

from cs336_basics.nn import Linear, softmax

def scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    d_k = Q.shape[-1]

    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    if mask is not None:
        mask = mask.to(device=scores.device, dtype=torch.bool)
        scores = scores.masked_fill(~mask, float("-inf"))

    probs = softmax(scores, dim=-1)

    return torch.matmul(probs, V)

class RotaryPositionalEmbedding(nn.Module):
    def __init__(
        self,
        theta: float,
        d_k: int,
        max_seq_len: int,
        device=None,
    ):
        super().__init__()

        if d_k % 2 != 0:
            raise ValueError("d_k must be even for RoPE.")

        self.theta = theta
        self.d_k = d_k
        self.max_seq_len = max_seq_len

        inv_freq = theta ** (
            -torch.arange(0, d_k, 2, device=device, dtype=torch.float32) / d_k
        )

        positions = torch.arange(
            max_seq_len,
            device=device,
            dtype=torch.float32,
        )

        angles = torch.outer(positions, inv_freq)

        self.register_buffer("cos", torch.cos(angles), persistent=False)
        self.register_buffer("sin", torch.sin(angles), persistent=False)

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor,
    ) -> torch.Tensor:
        token_positions = token_positions.to(device=x.device, dtype=torch.long)

        cos = self.cos[token_positions].to(dtype=x.dtype)
        sin = self.sin[token_positions].to(dtype=x.dtype)

        x_even = x[..., 0::2]
        x_odd = x[..., 1::2]

        while cos.ndim < x_even.ndim:
            cos = cos.unsqueeze(-3)
            sin = sin.unsqueeze(-3)

        rotated_even = x_even * cos - x_odd * sin
        rotated_odd = x_even * sin + x_odd * cos

        out = torch.empty_like(x)
        out[..., 0::2] = rotated_even
        out[..., 1::2] = rotated_odd

        return out

class MultiHeadSelfAttention(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        theta: float | None = None,
        max_seq_len: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads.")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        self.q_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.k_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.v_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )
        self.output_proj = Linear(
            in_features=d_model,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

        if theta is not None:
            if max_seq_len is None:
                raise ValueError("max_seq_len must be provided when using RoPE.")

            self.rope = RotaryPositionalEmbedding(
                theta=theta,
                d_k=self.head_dim,
                max_seq_len=max_seq_len,
                device=device,
            )
        else:
            self.rope = None

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        *leading_dims, seq_len, d_model = x.shape

        x = x.reshape(
            *leading_dims,
            seq_len,
            self.num_heads,
            self.head_dim,
        )

        return x.transpose(-3, -2)

    def _merge_heads(self, x: torch.Tensor) -> torch.Tensor:
        *leading_dims, num_heads, seq_len, head_dim = x.shape

        x = x.transpose(-3, -2)

        return x.reshape(
            *leading_dims,
            seq_len,
            num_heads * head_dim,
        )

    def forward(
        self,
        x: torch.Tensor,
        token_positions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        seq_len = x.shape[-2]

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = self._split_heads(q)
        k = self._split_heads(k)
        v = self._split_heads(v)

        if self.rope is not None:
            if token_positions is None:
                token_positions = torch.arange(
                    seq_len,
                    device=x.device,
                    dtype=torch.long,
                )

            q = self.rope(q, token_positions)
            k = self.rope(k, token_positions)

        causal_mask = torch.tril(
            torch.ones(
                seq_len,
                seq_len,
                device=x.device,
                dtype=torch.bool,
            )
        )

        attn_out = scaled_dot_product_attention(
            q,
            k,
            v,
            mask=causal_mask,
        )

        attn_out = self._merge_heads(attn_out)

        return self.output_proj(attn_out)
