from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator

import torch
from torch.profiler import record_function

import cs336_basics.model as model_module


@contextlib.contextmanager
def stage_range(
    name: str,
    *,
    device: torch.device,
    enabled: bool,
    use_nvtx: bool,
) -> Iterator[None]:
    if not enabled:
        yield
        return
    with record_function(name):
        if use_nvtx and device.type == "cuda":
            with torch.cuda.nvtx.range(name):
                yield
        else:
            yield


def _profiled_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None,
    *,
    device: torch.device,
    use_nvtx: bool,
) -> torch.Tensor:
    d_k = K.shape[-1]
    with stage_range("attention/scores", device=device, enabled=True, use_nvtx=use_nvtx):
        attention_scores = model_module.einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))
    with stage_range("attention/softmax", device=device, enabled=True, use_nvtx=use_nvtx):
        attention_weights = model_module.softmax(attention_scores, dim=-1)
    with stage_range("attention/value", device=device, enabled=True, use_nvtx=use_nvtx):
        return model_module.einsum(
            attention_weights,
            V,
            "... query key, ... key d_v -> ... query d_v",
        )


@contextlib.contextmanager
def instrument_attention(*, device: torch.device, enabled: bool, use_nvtx: bool) -> Iterator[None]:
    """Temporarily add ranges without modifying the upstream model implementation."""
    if not enabled:
        yield
        return
    original = model_module.scaled_dot_product_attention

    def profiled_attention(Q, K, V, mask=None):
        return _profiled_scaled_dot_product_attention(Q, K, V, mask, device=device, use_nvtx=use_nvtx)

    model_module.scaled_dot_product_attention = profiled_attention
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original
