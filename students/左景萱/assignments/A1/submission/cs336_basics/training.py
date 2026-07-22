"""Core optimization, data-loading, and checkpointing utilities."""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, TypeVar

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


_T = TypeVar("_T")
CheckpointDestination = str | os.PathLike[str] | BinaryIO | IO[bytes]


def softmax(inputs: Tensor, dim: int) -> Tensor:
    """Compute a numerically stable softmax along the selected dimension."""

    shifted = inputs - inputs.amax(dim=dim, keepdim=True)
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy loss for unnormalized class logits."""

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if inputs.shape[:-1] != targets.shape:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must equal inputs batch shape {tuple(inputs.shape[:-1])}"
        )

    maximum = inputs.amax(dim=-1, keepdim=True)
    log_normalizer = maximum.squeeze(-1) + torch.log(torch.exp(inputs - maximum).sum(dim=-1))
    target_logits = inputs.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the combined L2 norm of all present gradients in place."""

    if max_l2_norm < 0:
        raise ValueError(f"max_l2_norm must be non-negative, got {max_l2_norm}")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    per_gradient_norms = []
    for gradient in gradients:
        values = gradient.coalesce().values() if gradient.is_sparse else gradient
        per_gradient_norms.append(torch.linalg.vector_norm(values.detach(), ord=2))

    norm_device = per_gradient_norms[0].device
    total_norm = torch.linalg.vector_norm(
        torch.stack([norm.to(norm_device) for norm in per_gradient_norms]),
        ord=2,
    )
    if not bool(torch.isfinite(total_norm)):
        raise FloatingPointError("non-finite gradient norm")
    coefficient = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)

    with torch.no_grad():
        for gradient in gradients:
            gradient.mul_(coefficient.to(device=gradient.device, dtype=gradient.dtype))


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        if lr < 0:
            raise ValueError(f"learning rate must be non-negative, got {lr}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"beta1 must be in [0, 1), got {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"beta2 must be in [0, 1), got {betas[1]}")
        if eps < 0:
            raise ValueError(f"epsilon must be non-negative, got {eps}")
        if weight_decay < 0:
            raise ValueError(f"weight_decay must be non-negative, got {weight_decay}")

        defaults = {
            "lr": lr,
            "betas": betas,
            "eps": eps,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure: Callable[[], _T] | None = None) -> _T | None:
        """Perform one optimization step and return an optional closure loss."""

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
                gradient = parameter.grad
                if gradient is None:
                    continue
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

                parameter.mul_(1 - learning_rate * weight_decay)
                exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                bias_correction1 = 1 - beta1**step
                bias_correction2 = 1 - beta2**step
                step_size = learning_rate * math.sqrt(bias_correction2) / bias_correction1
                denominator = exp_avg_sq.sqrt().add_(epsilon)
                parameter.addcdiv_(exp_avg, denominator, value=-step_size)

        return loss


def get_lr_cosine_schedule(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Linear warmup followed by cosine decay and a constant floor."""

    if iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {iteration}")
    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError("cosine_cycle_iters must be at least warmup_iters")

    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration > cosine_cycle_iters:
        return min_learning_rate
    if cosine_cycle_iters == warmup_iters:
        return min_learning_rate

    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (1 + math.cos(math.pi * progress))


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Uniformly sample next-token prediction examples from a token array."""

    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")
    if len(dataset) <= context_length:
        raise ValueError(f"dataset must contain more than context_length={context_length} tokens; got {len(dataset)}")

    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    token_windows = np.asarray(dataset[starts[:, None] + offsets[None, :]])
    tokens = torch.as_tensor(token_windows, dtype=torch.long, device=device)
    return tokens[:, :-1], tokens[:, 1:]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    """Serialize model, optimizer, and completed-iteration state."""

    payload = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    if isinstance(out, (str, os.PathLike)):
        destination = os.fspath(out)
        temporary = f"{destination}.tmp-{os.getpid()}"
        try:
            torch.save(payload, temporary)
            os.replace(temporary, destination)
        finally:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
    else:
        torch.save(payload, out)


def load_checkpoint(
    src: CheckpointDestination,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a checkpoint and return its completed-iteration count."""

    first_tensor = next(iter(model.parameters()), None)
    if first_tensor is None:
        first_tensor = next(iter(model.buffers()), None)
    map_location = first_tensor.device if first_tensor is not None else torch.device("cpu")

    checkpoint = torch.load(src, map_location=map_location, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
