from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def softmax(inputs: Tensor, dim: int) -> Tensor:
    """Numerically stable softmax implemented from elementary tensor ops."""
    shifted = inputs - inputs.max(dim=dim, keepdim=True).values
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy from logits without materializing softmax probabilities."""
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match all non-vocabulary logits dimensions")

    max_logits = logits.max(dim=-1, keepdim=True).values
    log_normalizer = max_logits.squeeze(-1) + torch.log(
        torch.exp(logits - max_logits).sum(dim=-1)
    )
    target_logits = logits.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float) -> None:
    """Clip all available parameter gradients using one shared global L2 norm."""
    if max_l2_norm <= 0:
        raise ValueError("max_l2_norm must be positive")
    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    squared_norm = torch.zeros((), device=gradients[0].device, dtype=torch.float32)
    for gradient in gradients:
        squared_norm += gradient.detach().to(torch.float32).square().sum()
    total_norm = torch.sqrt(squared_norm)
    scale = torch.clamp(max_l2_norm / (total_norm + 1e-6), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(device=gradient.device, dtype=gradient.dtype))
