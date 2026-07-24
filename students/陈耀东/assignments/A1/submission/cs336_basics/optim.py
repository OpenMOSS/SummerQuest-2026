"""CS336 Assignment 1 的优化器与学习率调度脚手架。"""

from __future__ import annotations

import math
from collections.abc import Iterable

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """AdamW 优化器脚手架。

    目标：
        实现 decoupled weight decay 版本的 Adam。

    每个参数需要维护的状态：
        - step: 当前参数已经更新了多少步
        - exp_avg: 一阶动量 m
        - exp_avg_sq: 二阶动量 v

    单步更新的核心结构：
        1. 读取梯度 g。
        2. 更新 m = beta1 * m + (1 - beta1) * g。
        3. 更新 v = beta2 * v + (1 - beta2) * g^2。
        4. 做 bias correction。
        5. 用 Adam 方向更新参数。
        6. 单独应用 decoupled weight decay。

    注意：
        - 不要把 weight_decay 混进梯度里。
        - 测试允许匹配参考实现或 PyTorch AdamW。
        - state_dict / load_state_dict 由 torch.optim.Optimizer 基类处理。
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        """执行一次 AdamW 参数更新。"""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            for p in group["params"]:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p,memory_format=torch.preserve_format)
                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(g, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step

                step_size = lr / bias_correction1

                bias_correction2_sqrt = math.sqrt(bias_correction2)
                denom = exp_avg_sq.sqrt().div_(bias_correction2_sqrt).add_(eps)

                if weight_decay != 0:
                    p.mul_(1 - lr * weight_decay)

                p.addcdiv_(exp_avg, denom, value=-step_size)
        return loss

def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """带 linear warmup 的 cosine learning-rate schedule。

    分段规则：
        - it < warmup_iters：从 0 线性升到 max_learning_rate。
        - warmup_iters <= it <= cosine_cycle_iters：余弦衰减到 min_learning_rate。
        - it > cosine_cycle_iters：保持 min_learning_rate。

    测试会逐点比较前 25 个 learning rate。
    """

    if it < warmup_iters:
        if warmup_iters ==0:
            return max_learning_rate
        return (it /warmup_iters) * max_learning_rate
    elif it <= cosine_cycle_iters:
        if cosine_cycle_iters == warmup_iters:
            return min_learning_rate
        progress =(it - warmup_iters) / (cosine_cycle_iters - warmup_iters)

        return min_learning_rate +0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))
    else:
        return min_learning_rate
