import math
from collections.abc import Iterable

import torch
from torch import Tensor
from torch.optim import Optimizer


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters

    if it > cosine_cycle_iters:
        return min_learning_rate

    progress = (
        (it - warmup_iters)
        / (cosine_cycle_iters - warmup_iters)
    )

    cosine_value = 0.5 * (1 + math.cos(math.pi * progress))

    return (
        min_learning_rate
        + cosine_value
        * (max_learning_rate - min_learning_rate)
    )

class AdamW(Optimizer):
    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ) -> None:
        if lr < 0:
            raise ValueError("lr must be non-negative")

        if eps < 0:
            raise ValueError("eps must be non-negative")

        beta1, beta2 = betas

        if not 0 <= beta1 < 1:
            raise ValueError("beta1 must be in [0, 1)")

        if not 0 <= beta2 < 1:
            raise ValueError("beta2 must be in [0, 1)")

        if weight_decay < 0:
            raise ValueError("weight_decay must be non-negative")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }

        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None

        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for parameter in group["params"]:
                if parameter.grad is None:
                    continue

                gradient = parameter.grad

                if gradient.is_sparse:
                    raise RuntimeError(
                        "AdamW does not support sparse gradients"
                    )

                state = self.state[parameter]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1

                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(
                    gradient,
                    alpha=1 - beta1,
                )

                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                corrected_exp_avg = exp_avg / bias_correction1
                corrected_exp_avg_sq = exp_avg_sq / bias_correction2

                parameter.mul_(1 - lr * weight_decay)

                denominator = corrected_exp_avg_sq.sqrt().add_(eps)

                parameter.addcdiv_(
                    corrected_exp_avg,
                    denominator,
                    value=-lr,
                )

        return loss