from __future__ import annotations

import math
import torch
from einops import einsum
from cs336_basics.nn_utils import softmax

def scaled_dot_product_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
    d_k = K.shape[-1]
    attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
    if mask is not None:
        attention_scores = torch.where(mask, attention_scores, float("-inf"))
    attention_weights = softmax(attention_scores, dim=-1)
    return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")
