from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import softmax


@contextmanager
def stage_range(name: str, *, emit_nvtx: bool = False) -> Iterator[None]:
    """Create a torch.profiler range, optionally mirrored to CUDA NVTX."""

    nvtx_context = None
    if emit_nvtx and torch.cuda.is_available():
        nvtx_context = torch.cuda.nvtx.range(name)
        nvtx_context.__enter__()
    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if nvtx_context is not None:
            nvtx_context.__exit__(None, None, None)


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Mathematically equivalent attention with stable profiler sub-ranges."""

    d_k = K.shape[-1]
    with stage_range("attention/scores"):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key")
        attention_scores = attention_scores / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with stage_range("attention/softmax"):
        attention_weights = softmax(attention_scores, dim=-1)

    with stage_range("attention/value"):
        return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


@contextmanager
def patched_attention_ranges() -> Iterator[None]:
    """Temporarily replace the public attention helper without editing starter code."""

    original = basics_model.scaled_dot_product_attention
    setattr(basics_model, "scaled_dot_product_attention", annotated_scaled_dot_product_attention)
    try:
        yield
    finally:
        setattr(basics_model, "scaled_dot_product_attention", original)
