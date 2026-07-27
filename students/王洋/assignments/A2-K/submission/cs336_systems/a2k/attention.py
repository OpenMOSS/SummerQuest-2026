from __future__ import annotations

import math

import torch


def explicit_attention(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Explicit QK^T -> mask -> softmax -> PV attention baseline."""
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        query_positions = torch.arange(query.shape[-2], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        mask = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value)


def attention_with_lse(
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    is_causal: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    scale = 1.0 / math.sqrt(query.shape[-1])
    scores = torch.matmul(query, key.transpose(-2, -1)) * scale
    if is_causal:
        query_positions = torch.arange(query.shape[-2], device=query.device)
        key_positions = torch.arange(key.shape[-2], device=query.device)
        mask = query_positions[:, None] >= key_positions[None, :]
        scores = scores.masked_fill(~mask, -1e6)
    lse = torch.logsumexp(scores.float(), dim=-1)
    probabilities = torch.softmax(scores, dim=-1)
    return torch.matmul(probabilities, value), lse
