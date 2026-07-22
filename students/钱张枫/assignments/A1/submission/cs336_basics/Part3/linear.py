from __future__ import annotations

import math

import torch
from einops import einsum
from torch import nn


class Linear(nn.Module):
    """无偏置线性层，对输入张量的最后一维执行线性变换。"""

    in_features: int
    out_features: int
    weight: nn.Parameter

    def __init__(
        self,
        in_features: int,
        out_features: int,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()

        # 保留输入/输出维度，便于初始化和调试时检查模块配置。
        self.in_features = in_features
        self.out_features = out_features

        factory_kwargs = {"device": device, "dtype": dtype}
        # 按 PyTorch nn.Linear 的权重布局存储 W，形状为 (out_features, in_features)。
        # forward 时再使用 W.T 与输入相乘，而不是直接把 W.T 存成参数。
        self.weight = nn.Parameter(torch.empty((out_features, in_features), **factory_kwargs))
        self.reset_parameters()

    def reset_parameters(self) -> None:
        # 使用 Transformer 常见的 Xavier 风格标准差，并通过截断正态限制极端初始值。
        std = math.sqrt(2.0 / (self.in_features + self.out_features))
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=-3.0 * std, b=3.0 * std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 支持任意 batch 维度，只要求输入最后一维等于 in_features。
        return einsum(x, self.weight, "... in_features, out_features in_features -> ... out_features")
