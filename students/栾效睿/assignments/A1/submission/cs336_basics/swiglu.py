import torch
import torch.nn as nn
from .linear import Linear


class SwiGlu(nn.Module):

    def __init__(self, d_model: int, d_ff: int | None, device=None, dtype=None):
        super().__init__()
        self.d_ff = d_ff or self._cal_d_ff(d_model)
        self.w1 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )
        self.w2 = Linear(
            in_features=self.d_ff, out_features=d_model, device=device, dtype=dtype
        )
        self.w3 = Linear(
            in_features=d_model, out_features=self.d_ff, device=device, dtype=dtype
        )

    @staticmethod
    def _cal_d_ff(d_model: int) -> int:
        if d_model * 8 % (3 * 64) == 0:
            return d_model * 8 // 3
        else:
            return ((d_model * 8) // (3 * 64) + 1) * 64

    @staticmethod
    def siLU(x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(self.siLU(self.w1(x)) * (self.w3(x)))
