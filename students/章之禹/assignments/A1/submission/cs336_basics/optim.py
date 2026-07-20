"""Optimization utilities used by the A1 language-model training loop.

The implementations in this module intentionally avoid the corresponding
high-level PyTorch helpers.  In particular, :class:`AdamW` follows Algorithm 1
in the version 26.0.3 assignment handout: decoupled weight decay is applied
before updating the moment estimates, and bias correction is folded into the
step size.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Iterable
from typing import Any, overload

import torch
from torch import Tensor
from torch.optim import Optimizer


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy without using ``torch.nn.functional``.

    ``logits`` may have any number of leading dimensions; ``targets`` must have
    exactly those leading dimensions.  The final dimension of ``logits`` is
    interpreted as the class/vocabulary dimension.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if logits.shape[:-1] != targets.shape:
        raise ValueError(
            "targets must match all non-class dimensions of logits: "
            f"got logits {tuple(logits.shape)} and targets {tuple(targets.shape)}"
        )
    if logits.shape[-1] == 0:
        raise ValueError("the class dimension must be non-empty")
    if targets.numel() == 0:
        raise ValueError("cross-entropy is undefined for an empty target tensor")
    if targets.dtype not in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64):
        raise TypeError("targets must contain integer class indices")

    # Subtracting the row maximum keeps exp() finite even for very large
    # logits.  The detached max is mathematically equivalent and avoids
    # backpropagating through an unnecessary max path.
    row_max = logits.max(dim=-1, keepdim=True).values.detach()
    shifted = logits - row_max
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    shifted_target_logits = shifted.gather(dim=-1, index=targets.to(torch.long).unsqueeze(-1)).squeeze(-1)
    # Keeping both terms in the shifted coordinate system avoids subtracting
    # two large, nearly equal values after adding row_max back in.
    return (log_normalizer - shifted_target_logits).mean()


def global_gradient_norm(parameters: Iterable[torch.nn.Parameter]) -> Tensor:
    """Compute the global L2 norm over all present parameter gradients."""

    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return torch.tensor(0.0)

    # Accumulating the squared norm in float64 on CPU and float32 elsewhere
    # avoids fp16 overflow without moving large gradients between devices.
    device = grads[0].device
    accumulation_dtype = torch.float64 if device.type == "cpu" else torch.float32
    squared_norm = torch.zeros((), device=device, dtype=accumulation_dtype)
    for grad in grads:
        if grad.device != device:
            raise ValueError("all gradients must be on the same device")
        values = grad.detach().coalesce().values() if grad.is_sparse else grad.detach()
        squared_norm.add_(values.to(accumulation_dtype).square().sum())
    return squared_norm.sqrt()


@torch.no_grad()
def clip_gradients(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
    *,
    epsilon: float = 1e-6,
) -> None:
    """Clip all gradients together to a maximum global L2 norm.

    The scale is ``max_l2_norm / (norm + 1e-6)``, matching the assignment and
    PyTorch's default numerical-stability constant.  Parameters without a
    gradient are ignored.
    """

    if not math.isfinite(max_l2_norm) or max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be a positive finite number")
    if epsilon < 0 or not math.isfinite(epsilon):
        raise ValueError("epsilon must be a non-negative finite number")

    parameters = tuple(parameters)
    total_norm = global_gradient_norm(parameters)
    scale = torch.clamp(max_l2_norm / (total_norm + epsilon), max=1.0)
    for parameter in parameters:
        grad = parameter.grad
        if grad is not None:
            grad.mul_(scale.to(device=grad.device, dtype=grad.dtype))


class AdamW(Optimizer):
    """AdamW optimizer from the assignment handout (version 26.0.3).

    Weight decay is decoupled from the gradient and is applied *before* the
    moment-adjusted update.  Hyperparameters can be overridden independently
    in standard PyTorch parameter groups.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict[str, Any]],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0 or not math.isfinite(lr):
            raise ValueError(f"invalid learning rate: {lr}")
        if len(betas) != 2 or not all(0 <= beta < 1 for beta in betas):
            raise ValueError(f"betas must each be in [0, 1), got {betas}")
        if eps < 0 or not math.isfinite(eps):
            raise ValueError(f"invalid epsilon: {eps}")
        if weight_decay < 0 or not math.isfinite(weight_decay):
            raise ValueError(f"invalid weight decay: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @overload
    def step(self, closure: None = None) -> None: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one optimization step and optionally return ``closure`` loss."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = float(group["lr"])
            beta1, beta2 = group["betas"]
            eps = float(group["eps"])
            weight_decay = float(group["weight_decay"])

            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")
                if torch.is_complex(parameter) or torch.is_complex(grad):
                    raise RuntimeError("this AdamW implementation does not support complex parameters")

                state = self.state[parameter]
                if not state:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                state["step"] += 1
                step = int(state["step"])
                exp_avg: Tensor = state["exp_avg"]
                exp_avg_sq: Tensor = state["exp_avg_sq"]

                # Algorithm 1, line 8: use the unadjusted learning rate for the
                # decoupled decay, before the moment-adjusted update.
                if weight_decay != 0:
                    parameter.add_(parameter, alpha=-lr * weight_decay)

                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)

                adjusted_lr = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                denominator = exp_avg_sq.sqrt().add_(eps)
                parameter.addcdiv_(exp_avg, denominator, value=-adjusted_lr)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by one cosine decay and a constant tail."""

    if it < 0:
        raise ValueError("iteration must be non-negative")
    if warmup_iters < 0:
        raise ValueError("warmup_iters must be non-negative")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be at least warmup_iters")
    if min_learning_rate < 0 or max_learning_rate < min_learning_rate:
        raise ValueError("learning rates must satisfy 0 <= min_learning_rate <= max_learning_rate")

    if warmup_iters > 0 and it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (1 + math.cos(math.pi * progress)) * (
        max_learning_rate - min_learning_rate
    )


# Explicit aliases make the public intent discoverable and keep adapter code
# concise without hiding the real implementations in tests/adapters.py.
gradient_clipping = clip_gradients
cosine_learning_rate = get_lr_cosine_schedule
