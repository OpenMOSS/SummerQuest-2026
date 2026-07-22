from __future__ import annotations

from collections.abc import Iterable

import torch


def softmax(in_features: torch.Tensor, dim: int) -> torch.Tensor:
    shifted = in_features - torch.max(in_features, dim=dim, keepdim=True).values
    exp = torch.exp(shifted)
    return exp / torch.sum(exp, dim=dim, keepdim=True)


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    max_logits = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_logits
    logsumexp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_logits.squeeze(-1)
    correct_logits = inputs[
        torch.arange(inputs.shape[0], device=inputs.device),
        targets,
    ]
    return torch.mean(logsumexp - correct_logits)


def gradient_clipping(
    parameters: Iterable[torch.nn.Parameter],
    max_l2_norm: float,
) -> None:
    grads = [parameter.grad for parameter in parameters if parameter.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(
        sum(torch.sum(grad.detach() ** 2) for grad in grads)
    )
    if total_norm <= max_l2_norm:
        return

    scale = max_l2_norm / (total_norm + 1e-6)
    for grad in grads:
        grad.mul_(scale)
