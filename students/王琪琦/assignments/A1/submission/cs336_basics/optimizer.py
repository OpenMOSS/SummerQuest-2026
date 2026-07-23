from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor


class AdamW(torch.optim.Optimizer):
    """Adam with bias correction and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("AdamW betas must be in [0, 1)")
        if eps < 0 or weight_decay < 0:
            raise ValueError("eps and weight_decay must be non-negative")
        super().__init__(
            params,
            {
                "lr": lr,
                "betas": betas,
                "eps": eps,
                "weight_decay": weight_decay,
            },
        )

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: None = None) -> None: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            learning_rate = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                gradient = parameter.grad
                if gradient is None:
                    continue
                if gradient.is_sparse:
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

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)
                parameter.mul_(1 - learning_rate * weight_decay)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = learning_rate * math.sqrt(bias_correction2) / bias_correction1
                parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-step_size)

        return loss


def cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by cosine decay and a constant minimum."""
    if iteration < 0 or warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError("invalid iteration or schedule boundaries")
    if min_learning_rate < 0 or max_learning_rate < min_learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min <= max")

    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
