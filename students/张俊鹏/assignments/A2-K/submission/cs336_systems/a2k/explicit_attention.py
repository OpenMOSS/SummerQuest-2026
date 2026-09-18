from __future__ import annotations

import math

import torch


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Explicit QK^T, mask, softmax, PV attention baseline."""
    if q.ndim != 3 or k.ndim != 3 or v.ndim != 3:
        raise ValueError("q, k, and v must have shape [batch, sequence, head_dim]")
    if q.shape[0] != k.shape[0] or k.shape[:2] != v.shape[:2]:
        raise ValueError("q, k, and v have incompatible batch/sequence shapes")
    if q.shape[-1] != k.shape[-1] or k.shape[-1] != v.shape[-1]:
        raise ValueError("q, k, and v must have the same head dimension")

    scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
    if is_causal:
        n_queries = q.shape[-2]
        n_keys = k.shape[-2]
        mask = torch.arange(n_queries, device=q.device)[:, None] >= torch.arange(n_keys, device=q.device)[None, :]
        scores = scores.masked_fill(~mask, float("-inf"))
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, v)
