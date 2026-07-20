from __future__ import annotations

import math

import torch
from torch import Tensor, nn


class Linear(nn.Module):
    """Bias-free linear transformation with weights stored as (d_out, d_in)."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError("d_in and d_out must be positive")
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.d_in + self.d_out))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3 * std, b=3 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """Trainable token lookup table."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


def silu(x: Tensor) -> Tensor:
    return x * torch.sigmoid(x)


class RMSNorm(nn.Module):
    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.rsqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (x_float * rms).to(input_dtype)
        return normalized * self.weight


class SwiGLU(nn.Module):
    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Two-projection SiLU FFN used by the parameter-matched ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        matched_width = max(1, round(1.5 * d_ff / 64) * 64)
        self.w1 = Linear(d_model, matched_width, device=device, dtype=dtype)
        self.w2 = Linear(matched_width, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))
