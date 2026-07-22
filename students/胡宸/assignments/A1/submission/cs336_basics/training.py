from __future__ import annotations

import math
from collections.abc import Iterable

import numpy as np
import torch


def get_batch(dataset: np.ndarray, batch_size: int, context_length: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
    starts = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    offsets = starts[:, None] + np.arange(context_length + 1)[None, :]
    batch = torch.as_tensor(dataset[offsets], dtype=torch.long, device=device)
    return batch[:, :-1], batch[:, 1:]


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(grad.detach() * grad.detach()) for grad in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + 1e-6)
        for grad in grads:
            grad.mul_(scale)


def cosine_lr_schedule(
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
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coefficient = 0.5 * (1 + math.cos(math.pi * progress))
    return min_learning_rate + coefficient * (max_learning_rate - min_learning_rate)
