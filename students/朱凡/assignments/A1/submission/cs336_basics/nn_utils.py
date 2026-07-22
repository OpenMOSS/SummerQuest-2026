"""Losses and neural-network utility functions."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor, nn


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over every non-vocabulary dimension.

    This implementation uses the log-sum-exp identity directly and therefore
    stays stable even when logits have very large magnitudes.
    """

    if logits.ndim < 1:
        raise ValueError("logits must have at least one dimension")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match all non-vocabulary dimensions of logits")
    if targets.dtype not in (torch.int32, torch.int64):
        raise TypeError("targets must be an integer tensor")

    max_logits = logits.amax(dim=-1, keepdim=True)
    shifted_logits = logits - max_logits
    log_normalizer = shifted_logits.exp().sum(dim=-1).log()
    target_logits = shifted_logits.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


@torch.no_grad()
def clip_gradients(parameters: Iterable[nn.Parameter], max_l2_norm: float) -> None:
    """Clip the combined L2 norm of all available gradients in place."""

    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    # Accumulate in float32 so low-precision training does not overflow here.
    squared_norm = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        squared_norm.add_(gradient.detach().float().square().sum())
    total_norm = squared_norm.sqrt()
    scale = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))


# Assignment-language alias.
gradient_clipping = clip_gradients
