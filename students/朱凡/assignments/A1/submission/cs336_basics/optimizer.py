"""Optimization algorithms and learning-rate schedules."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import nn


class SGD(torch.optim.Optimizer):
    """The handout's SGD example with a ``1 / sqrt(t + 1)`` decay."""

    def __init__(self, params: Iterable[nn.Parameter] | Iterable[dict[str, Any]], lr: float = 1e-3) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        super().__init__(params, dict(lr=lr))

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: Callable[[], float] | None = None) -> float | None: ...

    def step(  # ty: ignore[invalid-method-override] -- matches torch's published overloads above
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                base_lr: float = group["lr"]
                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue
                    state = self.state[parameter]
                    step = int(state.get("step", 0))
                    parameter.add_(parameter.grad, alpha=-base_lr / math.sqrt(step + 1))
                    state["step"] = step + 1
        return loss


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("learning rate must be non-negative")
        if not 0 <= betas[0] < 1 or not 0 <= betas[1] < 1:
            raise ValueError("betas must lie in [0, 1)")
        if eps < 0:
            raise ValueError("eps must be non-negative")
        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")
        super().__init__(params, dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay))

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: Callable[[], float] | None = None) -> float | None: ...

    def step(  # ty: ignore[invalid-method-override] -- matches torch's published overloads above
        self, closure: Callable[[], float] | None = None
    ) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                lr: float = group["lr"]
                beta1, beta2 = group["betas"]
                eps: float = group["eps"]
                weight_decay: float = group["weight_decay"]

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
                    step: int = state["step"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Decoupled weight decay uses the uncorrected base learning rate.
                    parameter.mul_(1 - lr * weight_decay)
                    exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                    corrected_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                    parameter.addcdiv_(exp_avg, exp_avg_sq.sqrt().add_(eps), value=-corrected_lr)

        return loss


def cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by one cosine decay cycle."""

    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    if warmup_iters < 0 or cosine_cycle_iters < warmup_iters:
        raise ValueError("expected 0 <= warmup_iters <= cosine_cycle_iters")
    if min_learning_rate < 0 or max_learning_rate < min_learning_rate:
        raise ValueError("expected 0 <= min_learning_rate <= max_learning_rate")

    if iteration < warmup_iters:
        return (iteration / warmup_iters) * max_learning_rate
    if iteration > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


get_lr_cosine_schedule = cosine_learning_rate
