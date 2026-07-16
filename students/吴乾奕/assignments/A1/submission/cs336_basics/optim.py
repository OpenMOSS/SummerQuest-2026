"""Optimization utilities used by the language-model training loop."""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import Tensor
from torch.optim import Optimizer


class AdamW(Optimizer):
    """Adam with decoupled weight decay.

    This implementation follows the bias-corrected update from the assignment
    and deliberately does not delegate to :class:`torch.optim.AdamW`.
    """

    def __init__(
        self,
        params: Iterable[Tensor] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"invalid beta parameter at index 1: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"invalid weight_decay value: {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], Tensor] | None = None) -> Tensor | None:
        """Perform one AdamW update and optionally return ``closure``'s loss."""

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
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                state["step"] += 1
                step = state["step"]
                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]

                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step

                # Weight decay is applied directly to the parameter, separate
                # from the gradient moments (the defining difference in AdamW).
                parameter.mul_(1 - lr * weight_decay)

                denominator = exp_avg_sq.sqrt().div_(math.sqrt(bias_correction2)).add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-(lr / bias_correction1))

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return linear-warmup/cosine-decay learning rate at iteration ``it``."""

    if it < 0:
        raise ValueError("it must be non-negative")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be greater than or equal to warmup_iters")

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it >= cosine_cycle_iters:
        return min_learning_rate

    # Here cosine_cycle_iters > warmup_iters, so the denominator is nonzero.
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine_factor * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> float:
    """Clip all gradients using one shared L2 norm and return the pre-clip norm."""

    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return 0.0

    # Taking a norm of the individual norms avoids concatenating every gradient
    # into a potentially enormous temporary tensor.
    individual_norms = [torch.linalg.vector_norm(gradient.detach().float(), ord=2) for gradient in gradients]
    first_device = individual_norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(first_device) for norm in individual_norms]),
        ord=2,
    )
    clip_coefficient = max_l2_norm / (total_norm + 1e-6)
    clip_coefficient = torch.clamp(clip_coefficient, max=1.0)

    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(clip_coefficient.to(device=gradient.device, dtype=gradient.dtype))
    return float(total_norm.detach().cpu())
