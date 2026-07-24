from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    shifted = logits - logits.max(dim=-1, keepdim=True).values
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1))
    correct = shifted.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - correct).mean()


def gradient_clipping(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    parameters = [parameter for parameter in parameters if parameter.grad is not None]
    if not parameters:
        return
    total = torch.sqrt(sum(parameter.grad.detach().float().square().sum() for parameter in parameters))
    scale = min(1.0, max_l2_norm / (total.item() + 1e-6))
    for parameter in parameters:
        parameter.grad.mul_(scale)


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[Tensor, Tensor]:
    if dataset.ndim != 1 or len(dataset) <= context_length:
        raise ValueError("dataset must be one-dimensional and longer than context_length")
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    x = np.stack([dataset[start : start + context_length] for start in starts])
    y = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return torch.as_tensor(x, dtype=torch.long, device=device), torch.as_tensor(y, dtype=torch.long, device=device)


def cosine_schedule(it: int, max_lr: float, min_lr: float, warmup_iters: int, cosine_cycle_iters: int) -> float:
    if it < warmup_iters:
        return max_lr * it / warmup_iters
    if it > cosine_cycle_iters:
        return min_lr
    ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    return min_lr + 0.5 * (1 + math.cos(math.pi * ratio)) * (max_lr - min_lr)
