from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch.optim import Optimizer


class AdamW(Optimizer):
    """实现解耦 weight decay 与偏差校正的一阶 AdamW 优化器。"""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0.0:
            raise ValueError(f"invalid learning rate: {lr}")
        if eps < 0.0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0.0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], torch.Tensor] | None = None) -> torch.Tensor | None:
        """根据当前梯度更新所有参数，并可选地返回 closure 重新计算的损失。"""

        loss = None
        if closure is not None:
            # Optimizer.step 通常处于 no_grad 上下文，closure 需要显式重新启用梯度追踪。
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                # Weight decay 独立于梯度与动量估计，避免被 Adam 的自适应缩放影响。
                if weight_decay != 0.0:
                    parameter.mul_(1.0 - lr * weight_decay)

                # m_t 和 v_t 分别是一阶、二阶原始矩的指数滑动平均。
                exp_avg.lerp_(gradient, 1.0 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1.0 - beta2)

                bias_correction1 = 1.0 - beta1**step
                bias_correction2 = 1.0 - beta2**step
                step_size = lr / bias_correction1
                denominator = exp_avg_sq.sqrt().div_(bias_correction2**0.5).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss
