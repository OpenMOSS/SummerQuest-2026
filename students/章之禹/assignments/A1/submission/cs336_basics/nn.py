"""From-scratch neural-network building blocks used by the Transformer.

The assignment deliberately avoids PyTorch's ready-made layers.  The modules in
this file therefore only rely on ``torch.nn.Module``, ``torch.nn.Parameter``,
container classes, and the explicitly permitted initialization utilities.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor
from torch.nn import Module, Parameter
from torch.nn.init import trunc_normal_


class Linear(Module):
    """A bias-free linear transformation storing its weight as ``(out, in)``."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if in_features <= 0 or out_features <= 0:
            raise ValueError("in_features and out_features must both be positive")

        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.empty(out_features, in_features, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: Tensor) -> Tensor:
        return x @ self.weight.transpose(-1, -2)


class Embedding(Module):
    """An embedding lookup table with the embedding dimension stored last."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError("num_embeddings and embedding_dim must both be positive")

        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        trunc_normal_(self.weight, mean=0.0, std=1.0, a=-3.0, b=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(Module):
    """Root-mean-square normalization with a learned elementwise gain."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if eps < 0:
            raise ValueError("eps must be non-negative")

        self.d_model = d_model
        self.eps = eps
        self.weight = Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, x: Tensor) -> Tensor:
        input_dtype = x.dtype
        x_float = x.to(torch.float32)
        rms = torch.sqrt(x_float.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = x_float / rms
        return (normalized * self.weight.to(torch.float32)).to(input_dtype)


def silu(x: Tensor) -> Tensor:
    """Apply the SiLU (Swish) activation elementwise."""

    return x * torch.sigmoid(x)


class SwiGLU(Module):
    """The gated feed-forward network ``W2(SiLU(W1 x) * W3 x)``."""

    def __init__(
        self,
        d_model: int,
        d_ff: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.d_model = d_model
        self.d_ff = d_ff
        self.w1 = Linear(d_model, d_ff, device=device, dtype=dtype)
        self.w2 = Linear(d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)) * self.w3(x))


class SiLUFeedForward(Module):
    """An ungated SiLU FFN used by the assignment's architecture ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        hidden_dim = 4 * d_model if d_ff is None else d_ff
        self.d_model = d_model
        self.d_ff = hidden_dim
        self.w1 = Linear(d_model, hidden_dim, device=device, dtype=dtype)
        self.w2 = Linear(hidden_dim, d_model, device=device, dtype=dtype)

    def forward(self, x: Tensor) -> Tensor:
        return self.w2(silu(self.w1(x)))


def softmax(x: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax, returning zeros for an all-``-inf`` row.

    The all-masked behavior is useful for attention: a query with no allowed keys
    should contribute a zero vector rather than propagate ``NaN`` values.
    """

    if not x.is_floating_point():
        raise TypeError("softmax expects a floating-point tensor")

    maxima = x.amax(dim=dim, keepdim=True)
    has_positive_infinity = torch.isposinf(maxima)

    # A slice containing +inf assigns equal probability to all +inf entries.
    positive_infinity_mask = torch.isposinf(x)
    positive_infinity_count = positive_infinity_mask.sum(dim=dim, keepdim=True).clamp_min(1)
    positive_infinity_probs = positive_infinity_mask.to(x.dtype) / positive_infinity_count

    # Replacing exceptional maxima by zero makes all--inf slices exponentiate to
    # all zeros.  Slices containing +inf are handled by the branch above.
    safe_x = torch.where(has_positive_infinity, torch.zeros_like(x), x)
    safe_maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    exponentials = torch.exp(safe_x - safe_maxima)
    denominator = exponentials.sum(dim=dim, keepdim=True)
    safe_denominator = torch.where(denominator > 0, denominator, torch.ones_like(denominator))
    probabilities = exponentials / safe_denominator
    probabilities = torch.where(denominator > 0, probabilities, torch.zeros_like(probabilities))

    return torch.where(has_positive_infinity, positive_infinity_probs, probabilities)
