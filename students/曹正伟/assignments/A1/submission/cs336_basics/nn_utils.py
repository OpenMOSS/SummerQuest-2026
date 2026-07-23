"""Numerically stable neural-network utility functions."""

from __future__ import annotations

from collections.abc import Iterable

import torch


def softmax(x: torch.Tensor, dim: int) -> torch.Tensor:
    """Apply a numerically stable softmax along ``dim``."""
    shifted = x - x.amax(dim=dim, keepdim=True)
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Return the mean cross-entropy over all batch-like dimensions."""
    if inputs.shape[:-1] != targets.shape:
        raise ValueError(f"targets must have shape {inputs.shape[:-1]}, got {targets.shape}")

    maximum = inputs.amax(dim=-1, keepdim=True)
    shifted = inputs - maximum
    log_normalizer = torch.log(torch.exp(shifted).sum(dim=-1)) + maximum.squeeze(-1)
    target_logits = inputs.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return (log_normalizer - target_logits).mean()


@torch.no_grad()
def clip_gradients(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip the combined L2 norm of parameter gradients in place."""
    if max_l2_norm < 0:
        raise ValueError("max_l2_norm must be non-negative")

    gradients = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not gradients:
        return

    devices = {gradient.device for gradient in gradients}
    if len(devices) != 1:
        raise ValueError("all gradients must be on the same device")
    per_gradient_norms = torch.stack([torch.linalg.vector_norm(gradient.detach().float()) for gradient in gradients])
    total_norm = torch.linalg.vector_norm(per_gradient_norms)
    scale = torch.clamp(max_l2_norm / (total_norm + eps), max=1.0)
    for gradient in gradients:
        gradient.mul_(scale.to(dtype=gradient.dtype))
