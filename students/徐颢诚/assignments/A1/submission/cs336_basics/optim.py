"""作业中实现的优化器。"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch


class AdamW(torch.optim.Optimizer):
    """按作业第 4.3 节伪代码实现的 decoupled-weight-decay AdamW。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta values: {betas}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight decay value: {weight_decay}")
        super().__init__(params, {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay})

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """先施加 decoupled weight decay，再更新 moments 与参数。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            epsilon = group["eps"]
            weight_decay = group["weight_decay"]
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad

                parameter.mul_(1 - learning_rate * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                adjusted_learning_rate = learning_rate * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(epsilon)
                parameter.addcdiv_(exp_avg, denominator, value=-adjusted_learning_rate)
        return loss


def get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """按 PDF 4.4 实现 warm-up、余弦退火和退火后的恒定学习率。"""
    if iteration < warmup_iters:
        return iteration / warmup_iters * max_learning_rate
    if iteration <= cosine_cycle_iters:
        cosine_ratio = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + (1 + math.cos(math.pi * cosine_ratio)) / 2 * (
            max_learning_rate - min_learning_rate
        )
    return min_learning_rate


@torch.no_grad()
def clip_grad_norm_(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> torch.Tensor:
    """按全局 L2 范数原地裁剪一组参数的梯度。

    将所有存在的梯度视为一个长向量。若其 L2 范数超过
    ``max_l2_norm``，则以同一个比例缩放每个梯度，因此不会改变
    不同参数梯度之间的相对方向。
    """
    if max_l2_norm < 0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return torch.tensor(0.0)

    # ||g||_2 = sqrt(sum_i ||g_i||_2^2)，其中 g_i 是每个参数的梯度。
    total_norm = torch.stack([gradient.norm(2) for gradient in gradients]).norm(2)
    # 加上很小的 epsilon，避免总范数为 0 时除零；clamp 保证只缩小、不放大。
    clip_coefficient = (max_l2_norm / (total_norm + 1e-6)).clamp(max=1.0)
    for gradient in gradients:
        gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return total_norm
