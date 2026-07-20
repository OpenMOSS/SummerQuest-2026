import math
import torch
import torch.nn as nn


class Linear(nn.Module):
    def __init__(self, in_features, out_features, device=None, dtype=None):
        super().__init__()
        weight = torch.empty(out_features, in_features, device=device, dtype=dtype)

        nn.init.trunc_normal_(
            weight,
            mean=0.0,
            std=math.sqrt(2.0 / (in_features + out_features)),
            a=-3.0,
            b=3.0,
        )
        # self.W = weight
        self.weight = nn.Parameter(weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x @ self.weight.T
