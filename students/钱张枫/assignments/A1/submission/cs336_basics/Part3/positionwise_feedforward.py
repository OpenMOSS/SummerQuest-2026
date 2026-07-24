from __future__ import annotations

import math

import torch
from torch import nn

from cs336_basics.Part3.linear import Linear


def silu(x: torch.Tensor) -> torch.Tensor:
    """Apply the SiLU activation without using ``torch.nn.functional``."""

    return x * torch.sigmoid(x)


class SwiGLU(nn.Module):
    """Position-wise SwiGLU feed-forward network."""

    d_model: int
    d_ff: int
    w1: Linear
    w2: Linear
    w3: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if d_ff is not None and d_ff <= 0:
            raise ValueError("d_ff must be greater than zero.")

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else self._default_d_ff(d_model)

        # w1 和 w3 负责上投影到 hidden 维度，w2 负责投影回 d_model。
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)
        self.w3 = Linear(d_model, self.d_ff, device=device, dtype=dtype)

    @staticmethod
    def _default_d_ff(d_model: int) -> int:
        # 作业要求约为 8 / 3 * d_model，并取为 64 的倍数；向上取整避免 hidden 维度偏小。
        return 64 * math.ceil((8 * d_model / 3) / 64)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate = self.w1(x)
        return self.w2(silu(gate) * self.w3(x))


class SiLUFeedForward(nn.Module):
    """Non-gated SiLU FFN with a parameter-matched hidden dimension.

    A SwiGLU FFN has three projection matrices with approximately
    ``3 * d_model * swiglu_d_ff`` parameters. This two-projection FFN uses
    ``1.5 * swiglu_d_ff`` hidden units so the compared variants have nearly
    the same parameter count. When constructed directly without ``d_ff``, the
    assignment's ``4 * d_model`` baseline is used.
    """

    d_model: int
    d_ff: int
    w1: Linear
    w2: Linear

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        if d_model <= 0:
            raise ValueError("d_model must be greater than zero.")
        if d_ff is not None and d_ff <= 0:
            raise ValueError("d_ff must be greater than zero.")

        self.d_model = d_model
        self.d_ff = d_ff if d_ff is not None else 4 * d_model
        self.w1 = Linear(d_model, self.d_ff, device=device, dtype=dtype)
        self.w2 = Linear(self.d_ff, d_model, device=device, dtype=dtype)

    @staticmethod
    def matched_hidden_dim(swiglu_d_ff: int) -> int:
        """Return a two-matrix width matching a three-matrix SwiGLU FFN."""

        if swiglu_d_ff <= 0:
            raise ValueError("swiglu_d_ff must be greater than zero.")
        return (3 * swiglu_d_ff + 1) // 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(silu(self.w1(x)))
