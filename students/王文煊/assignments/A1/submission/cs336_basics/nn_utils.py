"""Loss, gradient clipping and other numeric helpers."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    """Mean cross-entropy over a batch.

    ``logits``: (batch, vocab_size) unnormalized scores.
    ``targets``: (batch,) integer class indices.
    Computed with the log-sum-exp trick for numerical stability.
    """
    logits = logits - torch.max(logits, dim=-1, keepdim=True).values
    log_sum_exp = torch.log(torch.sum(torch.exp(logits), dim=-1))
    target_logits = torch.gather(logits, -1, targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_sum_exp - target_logits)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    """Clip the combined gradient of ``parameters`` to L2 norm ``max_l2_norm`` in place."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    total_norm = torch.sqrt(sum(torch.sum(g.detach() ** 2) for g in grads))
    if total_norm > max_l2_norm:
        scale = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.detach().mul_(scale)
