from __future__ import annotations

import torch
from einops import reduce
from torch import nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization，仅按最后一维做归一化。"""

    d_model: int
    eps: float
    weight: nn.Parameter

    def __init__(
        self,
        d_model: int,
        eps: float = 1e-5,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        self.d_model = d_model
        self.eps = eps

        factory_kwargs = {"device": device, "dtype": dtype}
        # RMSNorm 只包含逐维缩放参数，不包含 bias；初始值为 1，表示不改变归一化后的尺度。
        self.weight = nn.Parameter(torch.ones(d_model, **factory_kwargs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 先升到 float32 计算均方根，避免低精度 dtype 在平方和均值时损失过多精度。
        original_dtype = x.dtype
        x_float = x.to(torch.float32)

        mean_square = reduce(x_float.square(), "... d_model -> ... 1", "mean")
        rms = torch.rsqrt(mean_square + self.eps)
        normalized = x_float * rms
        scaled = normalized * self.weight.to(torch.float32)

        return scaled.to(original_dtype)
