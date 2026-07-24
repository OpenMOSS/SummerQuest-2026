from __future__ import annotations

import math
from collections.abc import Callable, Iterable

import torch
from torch import Tensor, nn


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the warmup-plus-cosine learning rate for one optimizer step."""
    if it < 0:
        raise ValueError("it must be non-negative.")
    if min_learning_rate < 0 or max_learning_rate < 0:
        raise ValueError("Learning rates must be non-negative.")
    if min_learning_rate > max_learning_rate:
        raise ValueError("min_learning_rate cannot exceed max_learning_rate.")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative.")
    if cosine_cycle_iters <= warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than warmup_iters.")

    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it <= cosine_cycle_iters:
        cosine_progress = (it - warmup_iters) / (
            cosine_cycle_iters - warmup_iters
        )
        cosine_weight = 0.5 * (1 + math.cos(math.pi * cosine_progress))
        return min_learning_rate + cosine_weight * (
            max_learning_rate - min_learning_rate
        )

    return min_learning_rate


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[nn.Parameter],
    max_l2_norm: float,
    eps: float = 1e-6,
) -> Tensor:
    """Clip gradients in place and return their global norm before clipping."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive.")
    if eps < 0:
        raise ValueError("eps must be non-negative.")

    gradients = [
        parameter.grad
        for parameter in parameters
        if parameter.grad is not None
    ]
    if not gradients:
        return torch.zeros((), dtype=torch.float32)
    if any(gradient.is_sparse for gradient in gradients):
        raise RuntimeError("Gradient clipping does not support sparse gradients.")

    reference_device = gradients[0].device
    if any(gradient.device != reference_device for gradient in gradients):
        raise ValueError("All gradients must be on the same device.")

    squared_norm = torch.zeros(
        (),
        device=reference_device,
        dtype=torch.float32,
    )
    for gradient in gradients:
        squared_norm.add_(gradient.to(torch.float32).square().sum())

    total_norm = squared_norm.sqrt()
    if not torch.isfinite(total_norm):
        return total_norm
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(dtype=gradient.dtype))
    return total_norm


class AdamW(torch.optim.Optimizer):
    """AdamW with bias correction and decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

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
                if parameter.grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[parameter]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter)
                    state["exp_avg_sq"] = torch.zeros_like(parameter)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                gradient = parameter.grad

                parameter.mul_(1 - lr * weight_decay)

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(
                    gradient,
                    gradient,
                    value=1 - beta2,
                )

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = lr * math.sqrt(bias_correction2) / bias_correction1

                parameter.addcdiv_(
                    exp_avg,
                    exp_avg_sq.sqrt().add_(eps),
                    value=-step_size,
                )

        return loss
