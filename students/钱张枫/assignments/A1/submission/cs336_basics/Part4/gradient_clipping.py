from __future__ import annotations

import math
from collections.abc import Iterable

import torch


@torch.no_grad()
def clip_gradient_norm_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """原地裁剪所有参数梯度组成的全局 L2 norm。"""

    if not math.isfinite(max_l2_norm) or max_l2_norm < 0.0:
        raise ValueError("max_l2_norm must be finite and not negative.")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    gradient_device = gradients[0].device
    if any(gradient.device != gradient_device for gradient in gradients):
        raise ValueError("all gradients must be on the same device for global norm clipping.")

    # 全部平方和保留在设备端，只构建一个标量 tensor，避免每个参数一次的 host-device 同步。
    total_squared_norm = torch.zeros((), device=gradient_device, dtype=torch.float32)
    for gradient in gradients:
        total_squared_norm.add_(gradient.detach().to(dtype=torch.float32).square().sum())
    total_l2_norm = total_squared_norm.sqrt()

    # 始终乘以不大于 1 的设备端系数；原本较小的梯度只乘 1，不需要 Python 标量分支。
    clip_coefficient = torch.clamp(max_l2_norm / (total_l2_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient.to(dtype=gradient.dtype))
