"""Small neural-network building blocks used by the language model.

The layers in this module deliberately expose the same parameter layout as
``torch.nn`` (out features first for linear weights), while keeping the core
operations explicit for the assignment.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _truncated_normal_(parameter: Tensor, std: float) -> Tensor:
    """Initialize with a normal distribution truncated at three stddevs."""

    if std <= 0:
        raise ValueError("standard deviation must be positive")

    # Inverse-transform sampling keeps initialization from scratch rather than
    # delegating to torch.nn.init. The constants below are the standard-normal
    # CDF at -3 and +3 standard deviations.
    lower_cdf = 0.0013498980316301035
    upper_cdf = 0.9986501019683699
    with torch.no_grad():
        parameter.uniform_(2 * lower_cdf - 1, 2 * upper_cdf - 1)
        parameter.erfinv_().mul_(std * math.sqrt(2)).clamp_(-3 * std, 3 * std)
    return parameter


class Identity(nn.Module):
    """A from-scratch identity layer used by the no-normalization ablation."""

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs


class Linear(nn.Module):
    """A bias-free affine projection."""

    def __init__(self, in_features: int, out_features: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must be positive")
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        std = math.sqrt(2 / (in_features + out_features))
        _truncated_normal_(self.weight, std)

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.in_features:
            raise ValueError(f"expected final dimension {self.in_features}, got {inputs.shape[-1]}")
        return inputs @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """A table of token embeddings."""

    def __init__(self, num_embeddings: int, embedding_dim: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must be positive")
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _truncated_normal_(self.weight, 1.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        if token_ids.dtype not in (torch.int32, torch.int64):
            raise TypeError("embedding indices must be integer tensors")
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(self, d_model: int, eps: float = 1e-5, *, device=None, dtype=None) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps <= 0:
            raise ValueError("eps must be positive")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"expected final dimension {self.d_model}, got {inputs.shape[-1]}")

        # The reduction is intentionally performed in fp32 for low-precision
        # training, as specified by the assignment.
        input_dtype = inputs.dtype
        inputs_fp32 = inputs.float()
        rms = torch.rsqrt(inputs_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (inputs_fp32 * rms).to(input_dtype)
        return normalized * self.weight


def silu(inputs: Tensor) -> Tensor:
    """The SiLU activation, written in terms of elementary tensor operations."""

    return inputs * torch.sigmoid(inputs)


class SwiGLU(nn.Module):
    """A SwiGLU position-wise feed-forward network."""

    def __init__(self, d_model: int, d_ff: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)) * self.w3(inputs))


class SiLUFeedForward(nn.Module):
    """Two-projection SiLU FFN used by the matched-parameter ablation."""

    def __init__(self, d_model: int, d_ff: int, *, device=None, dtype=None) -> None:
        super().__init__()
        if d_model <= 0 or d_ff <= 0:
            raise ValueError("d_model and d_ff must be positive")
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)))


__all__ = ["Embedding", "Identity", "Linear", "RMSNorm", "SiLUFeedForward", "SwiGLU", "silu"]
