"""Deliberately partial A1 implementation used to exercise the grading flow.

Only the small set of functions in this module is complete.  The remaining
assignment adapters intentionally keep their starter ``NotImplementedError``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import Tensor


def linear(in_features: Tensor, weights: Tensor) -> Tensor:
    return torch.nn.functional.linear(in_features, weights)


def embedding(token_ids: Tensor, weights: Tensor) -> Tensor:
    return torch.nn.functional.embedding(token_ids, weights)


def silu(in_features: Tensor) -> Tensor:
    return torch.nn.functional.silu(in_features)


def softmax(in_features: Tensor, dim: int) -> Tensor:
    return torch.nn.functional.softmax(in_features, dim=dim)


def cross_entropy(inputs: Tensor, targets: Tensor) -> Tensor:
    return torch.nn.functional.cross_entropy(inputs, targets)


def clip_gradients(
    parameters: Iterable[torch.nn.Parameter], max_l2_norm: float
) -> None:
    torch.nn.utils.clip_grad_norm_(parameters, max_l2_norm)
