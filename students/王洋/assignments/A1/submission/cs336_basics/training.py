"""Core utilities for language-model training.

This module intentionally keeps the individual components independent of a
particular model or training script.  In addition to making the utilities easy
to test, that lets the same data loader, optimizer, schedule, and checkpoint
format be reused by later experiment scripts.
"""

from __future__ import annotations

import math
import os
from collections.abc import Callable, Iterable
from typing import IO, BinaryIO, overload

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def softmax(in_features: Tensor, dim: int) -> Tensor:
    """Compute a numerically stable softmax along ``dim``.

    Subtracting the maximum does not change the result, but ensures the largest
    argument passed to ``exp`` is zero and therefore prevents overflow.
    """

    input_dtype = in_features.dtype
    working = in_features.to(torch.float32) if input_dtype in (torch.float16, torch.bfloat16) else in_features
    shifted = working - working.max(dim=dim, keepdim=True).values
    exponentials = shifted.exp()
    probabilities = exponentials / exponentials.sum(dim=dim, keepdim=True)
    return probabilities.to(input_dtype) if input_dtype in (torch.float16, torch.bfloat16) else probabilities


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    """Return mean cross-entropy from unnormalized class logits.

    ``inputs`` may have any number of leading batch-like dimensions.  Its last
    dimension is interpreted as the class dimension and ``targets`` must have
    the corresponding leading shape.
    """

    if inputs.ndim < 1:
        raise ValueError("inputs must have at least one dimension")
    if targets.shape != inputs.shape[:-1]:
        raise ValueError(
            f"targets shape {tuple(targets.shape)} must equal inputs batch shape {tuple(inputs.shape[:-1])}"
        )

    # Work with shifted logits so neither the exponential nor the logarithm
    # sees the potentially very large original values.  Selecting from the
    # shifted logits also algebraically cancels the subtracted maximum.
    working = inputs.to(torch.float32) if inputs.dtype in (torch.float16, torch.bfloat16) else inputs
    shifted = working - working.max(dim=-1, keepdim=True).values
    target_logits = shifted.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    log_normalizer = shifted.exp().sum(dim=-1).log()
    return (log_normalizer - target_logits).mean()


@torch.no_grad()
def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip the global L2 norm of parameter gradients in place.

    Parameters without a gradient are ignored.  The ``1e-6`` term matches the
    numerical-stability constant prescribed by the assignment and used by
    PyTorch's default gradient clipping implementation.
    """

    if max_l2_norm <= 0:
        raise ValueError(f"max_l2_norm must be positive, got {max_l2_norm}")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    def gradient_norm(gradient: Tensor) -> Tensor:
        if gradient.is_sparse:
            gradient = gradient.coalesce().values()
        return torch.linalg.vector_norm(gradient.detach(), ord=2)

    first_device = gradients[0].device
    per_gradient_norms = [gradient_norm(gradient).to(first_device) for gradient in gradients]
    total_norm = torch.linalg.vector_norm(torch.stack(per_gradient_norms), ord=2)
    clip_coefficient = max_l2_norm / (total_norm + 1e-6)
    clip_coefficient = torch.clamp(clip_coefficient, max=1.0)

    for gradient in gradients:
        coefficient = clip_coefficient.to(device=gradient.device)
        if gradient.is_sparse:
            gradient._values().mul_(coefficient)
        else:
            gradient.mul_(coefficient)


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample next-token-prediction examples from a one-dimensional dataset.

    Exactly ``context_length + 1`` adjacent token IDs are read for each sample.
    This keeps the function efficient for memory-mapped arrays: only the tokens
    needed for the current batch are materialized in RAM.
    """

    if getattr(dataset, "ndim", None) != 1:
        raise ValueError("dataset must be a one-dimensional array of token IDs")
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")
    if context_length <= 0:
        raise ValueError(f"context_length must be positive, got {context_length}")

    num_starting_positions = len(dataset) - context_length
    if num_starting_positions <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens "
            f"(got {len(dataset)} tokens and context_length={context_length})"
        )

    starts = np.random.randint(0, num_starting_positions, size=batch_size)
    offsets = np.arange(context_length + 1)
    indices = starts[:, None] + offsets[None, :]
    # int64 gives the required torch.long token-ID representation regardless of
    # whether the backing array uses uint16, uint32, or another integer dtype.
    sequences = np.asarray(dataset[indices], dtype=np.int64)
    sequence_tensor = torch.from_numpy(sequences).to(device=device)
    return sequence_tensor[:, :-1], sequence_tensor[:, 1:]


class AdamW(torch.optim.Optimizer):
    """Adam with decoupled weight decay.

    The update follows Algorithm 1 in the assignment handout.  Moment tensors
    are initialized lazily so parameters that never receive gradients do not
    consume optimizer-state memory.
    """

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01,
    ) -> None:
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if len(betas) != 2:
            raise ValueError(f"betas must contain two values, got {betas}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
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

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        """Perform one AdamW update and optionally return the closure's loss."""

        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        with torch.no_grad():
            for group in self.param_groups:
                learning_rate = group["lr"]
                beta1, beta2 = group["betas"]
                epsilon = group["eps"]
                weight_decay = group["weight_decay"]

                for parameter in group["params"]:
                    if parameter.grad is None:
                        continue

                    gradient = parameter.grad
                    if gradient.is_sparse:
                        raise RuntimeError("AdamW does not support sparse gradients")

                    state = self.state[parameter]
                    if not state:
                        state["t"] = 0
                        state["exp_avg"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)
                        state["exp_avg_sq"] = torch.zeros_like(parameter, memory_format=torch.preserve_format)

                    state["t"] += 1
                    step = state["t"]
                    exp_avg = state["exp_avg"]
                    exp_avg_sq = state["exp_avg_sq"]

                    # Decoupled weight decay is based on the unadjusted learning
                    # rate and is applied independently of the moment update.
                    parameter.add_(parameter, alpha=-learning_rate * weight_decay)

                    exp_avg.mul_(beta1).add_(gradient, alpha=1 - beta1)
                    exp_avg_sq.mul_(beta2).addcmul_(gradient, gradient, value=1 - beta2)

                    adjusted_learning_rate = learning_rate * math.sqrt(1 - beta2**step) / (1 - beta1**step)
                    denominator = exp_avg_sq.sqrt().add_(epsilon)
                    parameter.addcdiv_(exp_avg, denominator, value=-adjusted_learning_rate)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    """Return the warmup-plus-cosine learning rate at iteration ``it``."""

    if warmup_iters < 0:
        raise ValueError(f"warmup_iters must be non-negative, got {warmup_iters}")
    if cosine_cycle_iters < warmup_iters:
        raise ValueError(f"cosine_cycle_iters must be at least warmup_iters, got {cosine_cycle_iters} < {warmup_iters}")

    if it < warmup_iters:
        return max_learning_rate * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_learning_rate

    # When both boundaries coincide there is no cosine interval.  The shared
    # endpoint is the maximum LR, followed immediately by post-annealing.
    if cosine_cycle_iters == warmup_iters:
        return max_learning_rate

    cosine_progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine_multiplier = 0.5 * (1 + math.cos(math.pi * cosine_progress))
    return min_learning_rate + cosine_multiplier * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike[str] | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer, and iteration state to a path or file object."""

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike[str] | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state and return the saved iteration."""

    parameter = next(model.parameters(), None)
    buffer = next(model.buffers(), None) if parameter is None else None
    model_device = (
        parameter.device if parameter is not None else buffer.device if buffer is not None else torch.device("cpu")
    )
    checkpoint = torch.load(src, map_location=model_device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


__all__ = [
    "AdamW",
    "cross_entropy",
    "get_batch",
    "get_lr_cosine_schedule",
    "gradient_clipping",
    "load_checkpoint",
    "save_checkpoint",
    "softmax",
]
