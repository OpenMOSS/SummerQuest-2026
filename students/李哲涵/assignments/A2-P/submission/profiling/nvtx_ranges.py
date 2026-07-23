from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch


@contextmanager
def profile_range(name: str) -> Iterator[None]:
    with torch.profiler.record_function(name):
        use_nvtx = torch.cuda.is_available()
        if use_nvtx:
            torch.cuda.nvtx.range_push(name)
        try:
            yield
        finally:
            if use_nvtx:
                torch.cuda.nvtx.range_pop()


def _profiled_scaled_dot_product_attention(Q, K, V, mask=None):
    from einops import einsum
    from cs336_basics.nn_utils import softmax

    with profile_range("attention/scores"):
        attention_scores = einsum(
            Q,
            K,
            "... query d_k, ... key d_k -> ... query key",
        ) / math.sqrt(K.shape[-1])
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with profile_range("attention/softmax"):
        attention_weights = softmax(attention_scores, dim=-1)
    with profile_range("attention/value"):
        return einsum(
            attention_weights,
            V,
            "... query key, ... key d_v -> ... query d_v",
        )


@contextmanager
def instrument_attention() -> Iterator[None]:
    import cs336_basics.model as model_module

    original = model_module.scaled_dot_product_attention
    model_module.scaled_dot_product_attention = _profiled_scaled_dot_product_attention
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original
