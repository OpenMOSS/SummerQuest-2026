import math
from collections.abc import Iterable, Callable
from typing import Any

import torch
from torch import Tensor
from torch.nn import Parameter
from torch.optim import Optimizer


class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        # 1. 检查超参数
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")

        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")

        if weight_decay < 0:
            raise ValueError(
                f"Invalid weight_decay value: {weight_decay}"
            )

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError(f"Invalid beta1 value: {beta1}")

        if not 0 <= beta2 < 1:
            raise ValueError(f"Invalid beta2 value: {beta2}")

        # 2. 交给 Optimizer 基类保存参数组与默认超参数
        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(
        self,
        closure: Callable[[], Tensor] | None = None,
    ) -> Tensor | None:
        """
        Perform one AdamW parameter update.
        """
        loss = None

        # 一般训练不会传 closure，但 Optimizer 接口允许它存在。
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        # Optimizer 支持多个 parameter group。
        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                # 当前参数没有梯度时，不更新，也不增加它自己的 step。
                if parameter.grad is None:
                    continue

                grad = parameter.grad

                if grad.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                # self.state 是由 Optimizer 基类提供的：
                # 每个参数都有自己独立的状态字典。
                state: dict[str, Any] = self.state[parameter]

                # 第一次遇到这个参数时初始化状态。
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                # 当前参数自己的更新次数。
                state["step"] += 1
                step = state["step"]

                # 3. Decoupled weight decay
                #
                # parameter <- parameter - lr * wd * parameter
                #
                # 等价于：
                # parameter <- (1 - lr * wd) * parameter
                if weight_decay != 0:
                    parameter.mul_(1 - lr * weight_decay)

                # 4. 更新一阶矩：
                # m <- beta1 * m + (1-beta1) * grad
                exp_avg.mul_(beta1).add_(
                    grad,
                    alpha=1 - beta1,
                )

                # 5. 更新二阶矩：
                # v <- beta2 * v + (1-beta2) * grad^2
                exp_avg_sq.mul_(beta2).addcmul_(
                    grad,
                    grad,
                    value=1 - beta2,
                )

                # 6. 偏置修正后的 learning rate
                #
                # alpha_t =
                # lr * sqrt(1-beta2^t) / (1-beta1^t)
                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                step_size = (
                    lr
                    * math.sqrt(bias_correction2)
                    / bias_correction1
                )

                # 7. Adam adaptive update：
                #
                # parameter <-
                # parameter - alpha_t * m / (sqrt(v) + eps)
                denominator = exp_avg_sq.sqrt().add_(eps)

                parameter.addcdiv_(
                    exp_avg,
                    denominator,
                    value=-step_size,
                )

        return loss