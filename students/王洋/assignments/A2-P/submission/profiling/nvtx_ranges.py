from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch


def profiled_scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = K.shape[-1]
    with torch.profiler.record_function("attention/scores"):
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)
        if mask is not None:
            scores = torch.where(mask, scores, float("-inf"))
    with torch.profiler.record_function("attention/softmax"):
        probabilities = torch.softmax(scores, dim=-1)
    with torch.profiler.record_function("attention/value"):
        return torch.matmul(probabilities, V)


@contextlib.contextmanager
def instrument_attention() -> Iterator[None]:
    import cs336_basics.model as model_module

    original = model_module.scaled_dot_product_attention
    model_module.scaled_dot_product_attention = profiled_scaled_dot_product_attention
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original
