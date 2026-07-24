from __future__ import annotations

import math
from contextlib import contextmanager
from typing import Callable

import torch
import torch.cuda.nvtx as torch_nvtx
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import softmax

try:
    import nvtx
except ImportError:  # pragma: no cover - depends on runtime environment
    nvtx = None

_ORIGINAL_SDPA: Callable | None = None
@contextmanager
def nvtx_range(name: str):
    with nvtx_range_in_domain("step", name):
        yield


@contextmanager
def nvtx_range_in_domain(domain_name: str, message: str):
    if nvtx is None:
        with torch_nvtx.range(f"{domain_name}/{message}"):
            yield
        return

    domain_ctor = getattr(nvtx, "Domain", None)
    if callable(domain_ctor):
        domain = domain_ctor(domain_name)
        annotate = getattr(domain, "annotate", None)
        if callable(annotate):
            with annotate(message):
                yield
            return

    annotate_fn = getattr(nvtx, "annotate", None)
    if callable(annotate_fn):
        with annotate_fn(message=message, domain=domain_name):
            yield
        return

    # Final fallback: keep the experiment runnable even if domain APIs are missing.
    with torch_nvtx.range(f"{domain_name}/{message}"):
        yield


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    with nvtx_range_in_domain("attention", "attention"):
        d_k = K.shape[-1]
        with nvtx_range_in_domain("attention", "scores"):
            attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)

        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with nvtx_range_in_domain("attention", "softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with nvtx_range_in_domain("attention", "value"):
            return einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")


def install_attention_nvtx() -> None:
    global _ORIGINAL_SDPA
    if _ORIGINAL_SDPA is None:
        _ORIGINAL_SDPA = basics_model.scaled_dot_product_attention
        basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def uninstall_attention_nvtx() -> None:
    global _ORIGINAL_SDPA
    if _ORIGINAL_SDPA is not None:
        basics_model.scaled_dot_product_attention = _ORIGINAL_SDPA
        _ORIGINAL_SDPA = None
