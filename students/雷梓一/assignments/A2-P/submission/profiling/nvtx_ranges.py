from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any, cast

import torch
from einops import einsum

import cs336_basics.model as basics_model

from .common import range_context


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Equivalent staff attention with ranges around its three expensive stages."""
    device = Q.device
    with range_context("attention/scores", device):
        scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(K.shape[-1])
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
    with range_context("attention/softmax", device):
        weights = basics_model.softmax(scores, dim=-1)
    with range_context("attention/value", device):
        return einsum(weights, V, "... query key, ... key d_v -> ... query d_v")


@contextlib.contextmanager
def annotated_attention() -> Iterator[None]:
    """Temporarily replace the module-global attention function used by the model."""
    original = basics_model.scaled_dot_product_attention
    basics_model.scaled_dot_product_attention = cast(Any, annotated_scaled_dot_product_attention)
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = original
