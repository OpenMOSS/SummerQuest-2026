"""NVTX-annotated drop-in replacement for the basics attention function."""

from __future__ import annotations

import math

import torch
from einops import einsum

from cs336_basics.nn_utils import softmax
from profiling.nvtx_ranges import nvtx_range


def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
    """Compute attention while exposing its three expensive phases to profilers."""
    d_k = K.shape[-1]
    with nvtx_range("attention/scores"):
        scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
    with nvtx_range("attention/softmax"):
        weights = softmax(scores, dim=-1)
    with nvtx_range("attention/value"):
        return einsum(weights, V, "... query key, ... key d_v -> ... query d_v")


def install_annotated_attention() -> None:
    """Patch the fixed starter model at its module-level lookup site."""
    import cs336_basics.model as basics_model

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


__all__ = ["annotated_scaled_dot_product_attention", "install_annotated_attention"]
