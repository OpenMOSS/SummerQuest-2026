"""CS336 Assignment 1 的神经网络工具函数脚手架。"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def softmax(x: Tensor, dim: int) -> Tensor:
    """数值稳定版 softmax。

    Shape 约定：
        x:   任意 shape
        out: 与 x 相同 shape

    数学目标：
        softmax(x_i) = exp(x_i) / sum_j exp(x_j)

    数值稳定技巧：
        - 先沿 dim 减去 max(x)，再做 exp。
        - 减 max 不改变 softmax 结果，但能避免 exp 溢出。

    测试会检查：
        - 与 torch.nn.functional.softmax 对齐。
        - 输入整体加上 100 后仍然稳定。
    """
    max_x = torch.max(x, dim=dim, keepdim=True).values
    x_stable = x - max_x
    exp_x = torch.exp(x_stable)
    return exp_x / torch.sum(exp_x, dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """平均 cross entropy loss。

    Shape 约定：
        inputs:  (batch_size, vocab_size)，未归一化 logits
        targets: (batch_size,)，每个元素是正确类别 ID
        out:     标量张量

    数学目标：
        loss_i = -log softmax(inputs_i)[targets_i]
        loss = mean_i loss_i

    数值稳定路线：
        - 使用 log-sum-exp 技巧：
          logsumexp(x) = m + log(sum(exp(x - m)))
        - 不要先显式 softmax 再 log，这在大 logits 下不稳定。

    测试会检查：
        - 与 torch.nn.functional.cross_entropy 对齐。
        - 输入乘以 1000 后仍然稳定。
    """
    max_val = inputs.max(dim=1, keepdim=True).values
    shifted_inputs = inputs - max_val

    log_sum_exp = max_val.squeeze(1) + torch.log(torch.exp(shifted_inputs).sum(dim=1))
    batch_indices = torch.arange(inputs.shape[0], device=inputs.device)
    correct_logits = inputs[batch_indices, targets]
    return (log_sum_exp - correct_logits).mean()


def gradient_clipping_with_norm(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
) -> torch.Tensor:
    """裁剪全局梯度并返回裁剪前的 L2 norm。"""
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    total_squared = torch.zeros((), device=grads[0].device, dtype=torch.float32)
    for grad in grads:
        total_squared.add_(grad.detach().float().pow(2).sum())
    total_norm = torch.sqrt(total_squared)

    if total_norm > max_l2_norm:
        clip_coefficient = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(clip_coefficient.to(dtype=grad.dtype))
    return total_norm


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """按全局 L2 norm 裁剪梯度，原地修改 parameter.grad。

    目标：
        如果所有非空梯度拼起来的总 L2 norm 超过 max_l2_norm，
        就把每个梯度都乘上同一个缩放系数。

    注意：
        - requires_grad=False 或 grad is None 的参数要跳过。
        - 这个函数不返回张量，直接原地改 grad。
        - 测试会与 torch.nn.utils.clip_grad_norm_ 的结果比较。

    推荐路线：
        1. 收集所有 p.grad is not None 的梯度。
        2. 计算 total_norm = sqrt(sum(grad.pow(2).sum()))。
        3. 如果 total_norm > max_l2_norm，则按 max_l2_norm / total_norm 缩放所有梯度。
    """
    gradient_clipping_with_norm(parameters, max_l2_norm)
