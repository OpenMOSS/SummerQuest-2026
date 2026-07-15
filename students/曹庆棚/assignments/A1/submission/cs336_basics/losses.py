from __future__ import annotations

import torch
from torch import Tensor


def softmax(x: Tensor, dim: int) -> Tensor:
    maximum = x.max(dim=dim, keepdim=True).values
    shifted = x - maximum
    exponentials = torch.exp(shifted)
    return exponentials / exponentials.sum(dim=dim, keepdim=True)


def cross_entropy(logits: Tensor, targets: Tensor) -> Tensor:
    if logits.ndim < 2:
        raise ValueError("logits must have at least two dimensions")
    if logits.shape[:-1] != targets.shape:
        raise ValueError("targets must match every logits dimension except the vocabulary dimension")

    maximum = logits.max(dim=-1, keepdim=True).values
    shifted = logits - maximum
    log_partition = torch.log(torch.exp(shifted).sum(dim=-1)) + maximum.squeeze(-1)
    target_logits = logits.gather(dim=-1, index=targets.long().unsqueeze(-1)).squeeze(-1)
    return (log_partition - target_logits).mean()
