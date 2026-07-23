"""From-scratch neural-network building blocks used by the Transformer.

Only tensor operations and the basic :mod:`torch.nn` container/parameter types are
used here.  In particular, the implementation deliberately does not delegate to
``nn.Linear``, ``nn.Embedding``, normalization layers, or packaged activations.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn


def _truncated_normal_(
    tensor: Tensor,
    *,
    mean: float = 0.0,
    std: float = 1.0,
    lower: float | None = None,
    upper: float | None = None,
) -> Tensor:
    """Fill ``tensor`` from a normal distribution truncated to an interval.

    This is an inverse-CDF implementation, so values are sampled from the actual
    truncated distribution rather than sampled normally and then clipped.
    """

    if std <= 0:
        raise ValueError(f"std must be positive, got {std}")
    lower = mean - 3 * std if lower is None else lower
    upper = mean + 3 * std if upper is None else upper
    if lower >= upper:
        raise ValueError(f"lower must be less than upper, got [{lower}, {upper}]")

    def normal_cdf(value: float) -> float:
        return (1.0 + math.erf((value - mean) / (std * math.sqrt(2.0)))) / 2.0

    cdf_lower = normal_cdf(lower)
    cdf_upper = normal_cdf(upper)
    work_dtype = tensor.dtype if tensor.dtype in (torch.float32, torch.float64) else torch.float32

    with torch.no_grad():
        samples = torch.empty(tensor.shape, device=tensor.device, dtype=work_dtype)
        samples.uniform_(2.0 * cdf_lower - 1.0, 2.0 * cdf_upper - 1.0)
        samples.erfinv_().mul_(std * math.sqrt(2.0)).add_(mean)
        samples.clamp_(min=lower, max=upper)
        tensor.copy_(samples)
    return tensor


class Linear(nn.Module):
    """A bias-free linear transformation with weight shape ``(d_out, d_in)``."""

    def __init__(
        self,
        d_in: int,
        d_out: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_in <= 0 or d_out <= 0:
            raise ValueError(f"linear dimensions must be positive, got d_in={d_in}, d_out={d_out}")
        self.d_in = d_in
        self.d_out = d_out
        self.weight = nn.Parameter(torch.empty(d_out, d_in, device=device, dtype=dtype))

        std = math.sqrt(2.0 / (d_in + d_out))
        _truncated_normal_(self.weight, std=std, lower=-3.0 * std, upper=3.0 * std)

    @property
    def in_features(self) -> int:
        return self.d_in

    @property
    def out_features(self) -> int:
        return self.d_out

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_in:
            raise ValueError(f"expected last input dimension {self.d_in}, got {inputs.shape[-1]}")
        return inputs @ self.weight.transpose(-1, -2)


class Embedding(nn.Module):
    """A learnable lookup table with shape ``(num_embeddings, embedding_dim)``."""

    def __init__(
        self,
        num_embeddings: int,
        embedding_dim: int,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if num_embeddings <= 0 or embedding_dim <= 0:
            raise ValueError(
                "embedding dimensions must be positive, "
                f"got num_embeddings={num_embeddings}, embedding_dim={embedding_dim}"
            )
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.weight = nn.Parameter(torch.empty(num_embeddings, embedding_dim, device=device, dtype=dtype))
        _truncated_normal_(self.weight, std=1.0, lower=-3.0, upper=3.0)

    def forward(self, token_ids: Tensor) -> Tensor:
        return self.weight[token_ids]


class RMSNorm(nn.Module):
    """Root-mean-square normalization over the final tensor dimension."""

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        if d_model <= 0:
            raise ValueError(f"d_model must be positive, got {d_model}")
        if eps <= 0:
            raise ValueError(f"eps must be positive, got {eps}")
        self.d_model = d_model
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model, device=device, dtype=dtype))

    def forward(self, inputs: Tensor) -> Tensor:
        if inputs.shape[-1] != self.d_model:
            raise ValueError(f"expected last input dimension {self.d_model}, got {inputs.shape[-1]}")

        # RMS accumulation is intentionally performed in fp32 even when the
        # surrounding model runs in fp16/bf16.
        inputs_fp32 = inputs.to(torch.float32)
        inverse_rms = torch.rsqrt(inputs_fp32.square().mean(dim=-1, keepdim=True) + self.eps)
        normalized = (inputs_fp32 * inverse_rms).to(inputs.dtype)
        return normalized * self.weight.to(dtype=inputs.dtype)


def silu(inputs: Tensor) -> Tensor:
    """Elementwise SiLU, written directly as ``x * sigmoid(x)``."""

    return inputs * torch.sigmoid(inputs)


class SiLU(nn.Module):
    """Module wrapper around :func:`silu`."""

    def forward(self, inputs: Tensor) -> Tensor:
        return silu(inputs)


class SwiGLU(nn.Module):
    """Gated feed-forward network ``W2(SiLU(W1(x)) * W3(x))``."""

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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)) * self.w3(inputs))


class SiLUFeedForward(nn.Module):
    """Ungated two-layer SiLU FFN used by the architecture ablation."""

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

    def forward(self, inputs: Tensor) -> Tensor:
        return self.w2(silu(self.w1(inputs)))


# A descriptive alias used by some training/configuration code.
SwiGLUFeedForward = SwiGLU


__all__ = [
    "Embedding",
    "Linear",
    "RMSNorm",
    "SiLU",
    "SiLUFeedForward",
    "SwiGLU",
    "SwiGLUFeedForward",
    "silu",
]
