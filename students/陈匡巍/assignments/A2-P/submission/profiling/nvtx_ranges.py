"""Attention-stage annotations shared by torch.profiler and Nsight Systems."""

from __future__ import annotations

import math

import torch
from torch.profiler import record_function


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with record_function("attention/scores"):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
    with record_function("attention/softmax"):
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
        probabilities = torch.softmax(scores, dim=-1)
    with record_function("attention/value"):
        return torch.matmul(probabilities, V)


def install_attention_ranges() -> None:
    import cs336_basics.model as model_module

    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention
