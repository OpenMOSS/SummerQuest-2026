"""Explicit PyTorch attention: QK^T, scale, causal mask, softmax, PV.

No fused attention (F.scaled_dot_product_attention / flash-attn / xFormers) is used.
"""

from __future__ import annotations

import math

import torch
from einops import einsum


def explicit_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    is_causal: bool = False,
) -> torch.Tensor:
    """Scaled dot-product attention written out explicitly.

    Args:
        q: (..., n_queries, d)
        k: (..., n_keys, d)
        v: (..., n_keys, d)
        is_causal: whether to apply a causal (lower-triangular) mask.
    """
    d = q.shape[-1]
    scale = 1.0 / math.sqrt(d)
    scores = einsum(q, k, "... q d, ... k d -> ... q k") * scale
    if is_causal:
        n_queries, n_keys = scores.shape[-2], scores.shape[-1]
        q_idx = torch.arange(n_queries, device=scores.device)[:, None]
        k_idx = torch.arange(n_keys, device=scores.device)[None, :]
        scores = scores.masked_fill(k_idx > q_idx, float("-inf"))
    probs = torch.softmax(scores, dim=-1)
    return einsum(probs, v, "... q k, ... k d -> ... q d")
