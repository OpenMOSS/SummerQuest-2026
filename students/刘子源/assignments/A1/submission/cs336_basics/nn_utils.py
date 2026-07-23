from __future__ import annotations

from collections.abc import Iterable

import torch


def softmax(inputs: torch.Tensor, dim: int) -> torch.Tensor:
    # 先减最大值，不然 exp 一激动就上天了。
    minus_max = inputs - inputs.max(dim=dim, keepdim=True).values
    exp_num = minus_max.exp()
    return exp_num / exp_num.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    biggest = inputs.max(dim=-1, keepdim=True).values
    log_sum = biggest.squeeze(-1) + torch.log(torch.exp(inputs - biggest).sum(dim=-1))
    right_answer = inputs.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum - right_answer).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")

    grad_list = [param.grad for param in parameters if param.grad is not None]
    if not grad_list:
        return

    total_norm = torch.stack([grad.detach().float().square().sum() for grad in grad_list]).sum().sqrt()
    shrink = (max_l2_norm / (total_norm + eps)).clamp(max=1.0)
    for grad in grad_list:
        # 没超就乘 1，超了再往回拽，省得再套一层 if。
        grad.mul_(shrink.to(device=grad.device, dtype=grad.dtype))
