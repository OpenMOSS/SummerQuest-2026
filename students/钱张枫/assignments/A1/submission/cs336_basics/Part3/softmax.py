from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """在指定维度上计算数值稳定的 softmax。"""

    # 先减去目标维度上的最大值，避免 exp 在大输入下溢出为 inf。
    shifted = x - torch.max(x, dim=dim, keepdim=True).values
    exp_shifted = torch.exp(shifted)
    return exp_shifted / torch.sum(exp_shifted, dim=dim, keepdim=True)
