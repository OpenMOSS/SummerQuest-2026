"""Shared NVTX and ``torch.profiler`` annotations for profiling runs.

The context manager in this module deliberately emits both kinds of range:
``torch.profiler.record_function`` makes the ranges visible in Chrome/Perfetto
traces, while NVTX makes the same hierarchy available to Nsight Systems.  NVTX
is best-effort so that the benchmark remains usable with ``--device cpu``.
"""

from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

import torch


@contextmanager
def profile_range(name: str) -> Iterator[None]:
    """Create a profiler range and, when CUDA is initialized, an NVTX range.

    ``torch.cuda.nvtx`` is unavailable in some CPU-only PyTorch builds.  A
    missing NVTX implementation must not prevent CPU smoke tests or
    ``torch.profiler`` CPU traces from running, so NVTX errors are contained
    here while ``record_function`` is always emitted.
    """

    nvtx_pushed = False
    if torch.cuda.is_initialized():
        try:
            torch.cuda.nvtx.range_push(name)
            nvtx_pushed = True
        except (AttributeError, RuntimeError):
            # CPU-only builds and builds without NVTX support reach this path.
            pass

    try:
        with torch.profiler.record_function(name):
            yield
    finally:
        if nvtx_pushed:
            try:
                torch.cuda.nvtx.range_pop()
            except (AttributeError, RuntimeError):
                pass


# Short alias useful when invoking this module directly from an Nsight-focused
# script.  Keep ``profile_range`` as the canonical name used by benchmark.py.
nvtx_range = profile_range


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Reference attention with ranges required by the profiling assignment.

    The signature matches ``cs336_basics.model.scaled_dot_product_attention``.
    ``torch.matmul`` is used rather than the original ``einx.einsum`` calls so
    the implementation works for every leading batch dimension accepted by the
    original function while keeping each logical attention stage explicit.
    """

    # Import lazily to avoid importing the assignment-1 package for callers who
    # only want generic profile ranges.
    from cs336_basics.nn_utils import softmax

    with profile_range("attention"):
        with profile_range("attention/scores"):
            attention_scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(K.shape[-1])
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))

        with profile_range("attention/softmax"):
            attention_weights = softmax(attention_scores, dim=-1)

        with profile_range("attention/value"):
            return torch.matmul(attention_weights, V)


@contextmanager
def patched_attention_ranges() -> Iterator[None]:
    """Temporarily install annotated attention into the basics Transformer.

    The assignment reference model resolves ``scaled_dot_product_attention``
    from the ``cs336_basics.model`` module at call time, so replacing that
    module attribute covers every attention layer without modifying upstream
    source code.  The original implementation is restored even if a benchmark
    or profiler run raises an exception.
    """

    import cs336_basics.model as basics_model

    original_attention: Any = basics_model.scaled_dot_product_attention
    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = original_attention
