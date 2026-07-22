import torch
import torch.nn as nn

from .linear import Linear


class SiLUFFN(nn.Module):
    """Two-projection SiLU feed-forward network used for the SwiGLU ablation."""

    def __init__(
        self,
        d_model: int,
        d_ff: int | None = None,
        device=None,
        dtype=None,
    ):
        super().__init__()
        self.d_ff = 4 * d_model if d_ff is None else d_ff
        if self.d_ff <= 0:
            raise ValueError(f"d_ff must be greater than 0, got {self.d_ff}")

        self.w1 = Linear(
            in_features=d_model,
            out_features=self.d_ff,
            device=device,
            dtype=dtype,
        )
        self.w2 = Linear(
            in_features=self.d_ff,
            out_features=d_model,
            device=device,
            dtype=dtype,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.w1(x)
        return self.w2(hidden * torch.sigmoid(hidden))
