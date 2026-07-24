from __future__ import annotations

import math
from collections.abc import Iterable
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor, nn

from cs336_basics.model import softmax


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more tokens than context_length")
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = np.arange(context_length + 1)
    windows = np.asarray(dataset[starts[:, None] + offsets[None, :]], dtype=np.int64)
    batch = torch.from_numpy(windows).to(device)
    return batch[:, :-1], batch[:, 1:]


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    maxima = torch.max(logits, dim=-1).values
    shifted = logits - maxima.unsqueeze(-1)
    log_normalizers = maxima + torch.log(torch.sum(torch.exp(shifted), dim=-1))
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_normalizers - target_logits)


def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return
    total_squared_norm = sum(gradient.detach().float().square().sum() for gradient in gradients)
    total_norm = torch.sqrt(total_squared_norm)
    scale = max_l2_norm / (total_norm + 1e-6)
    if scale < 1:
        for gradient in gradients:
            gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))


def cosine_learning_rate(
    iteration: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if iteration < warmup_iters:
        return max_learning_rate * iteration / warmup_iters
    if iteration > cosine_cycle_iters:
        return min_learning_rate
    progress = (iteration - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | bytes | IO[bytes] | BinaryIO,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | bytes | IO[bytes] | BinaryIO,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


__all__ = [
    "clip_gradients",
    "cosine_learning_rate",
    "cross_entropy",
    "get_batch",
    "load_checkpoint",
    "save_checkpoint",
    "softmax",
]
