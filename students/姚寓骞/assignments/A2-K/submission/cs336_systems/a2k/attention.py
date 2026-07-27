"""Explicit, unfused PyTorch attention baseline required by A2-K."""

from __future__ import annotations

import math

import torch


def explicit_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, is_causal: bool = False) -> torch.Tensor:
    """Compute QK^T, mask, softmax, and PV without fused attention APIs."""
    scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(q.shape[-1])
    if is_causal:
        n_queries, n_keys = q.shape[-2], k.shape[-2]
        q_pos = torch.arange(n_queries, device=q.device)[:, None]
        k_pos = torch.arange(n_keys, device=q.device)[None, :]
        scores = scores.masked_fill(q_pos < k_pos, float("-inf"))
    probabilities = torch.softmax(scores.float(), dim=-1).to(v.dtype)
    return torch.matmul(probabilities, v)


__all__ = ["explicit_attention"]
