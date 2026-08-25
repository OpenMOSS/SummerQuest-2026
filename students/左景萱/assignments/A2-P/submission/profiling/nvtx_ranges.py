"""Shared A2-P profiling annotations.

The same labels are emitted as ``record_function`` annotations for
``torch.profiler`` and as NVTX ranges on CUDA for Nsight Systems.  Attention
instrumentation is installed only for the duration of a benchmark run and is
restored afterwards.
"""

from __future__ import annotations

import contextlib
import math
from collections.abc import Iterator
from typing import Any

import torch
from einops import einsum

import cs336_basics.model as basics_model
from cs336_basics.nn_utils import softmax


PROFILE_WARMUP = "profile/warmup"
PROFILE_MEASURE = "profile/measure"
FORWARD = "forward"
BACKWARD = "backward"
OPTIMIZER = "optimizer"
ATTENTION_SCORES = "attention/scores"
ATTENTION_SOFTMAX = "attention/softmax"
ATTENTION_VALUE = "attention/value"

REQUIRED_TRACE_LABELS = (
    PROFILE_WARMUP,
    PROFILE_MEASURE,
    FORWARD,
    BACKWARD,
    OPTIMIZER,
    ATTENTION_SCORES,
    ATTENTION_SOFTMAX,
    ATTENTION_VALUE,
)


@contextlib.contextmanager
def annotated_range(
    label: str,
    *,
    device: torch.device,
    record_function: bool,
    nvtx: bool = True,
) -> Iterator[None]:
    """Emit one logical range without pretending that it owns worker kernels."""

    with contextlib.ExitStack() as stack:
        if record_function:
            stack.enter_context(torch.profiler.record_function(label))
        if nvtx and device.type == "cuda":
            stack.enter_context(torch.cuda.nvtx.range(label))
        yield


@contextlib.contextmanager
def instrument_attention(
    *,
    device: torch.device,
    record_function: bool,
    nvtx: bool = True,
) -> Iterator[None]:
    """Temporarily annotate the three required attention sub-operations.

    The replacement follows the starter implementation exactly: score matmul,
    optional masking, softmax, then value matmul.  The module global is restored
    even when the benchmark raises.
    """

    original = basics_model.scaled_dot_product_attention

    def annotated_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        with annotated_range(
            ATTENTION_SCORES,
            device=device,
            record_function=record_function,
            nvtx=nvtx,
        ):
            scores = einsum(
                Q,
                K,
                "... query d_k, ... key d_k -> ... query key",
            ) / math.sqrt(K.shape[-1])
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
        with annotated_range(
            ATTENTION_SOFTMAX,
            device=device,
            record_function=record_function,
            nvtx=nvtx,
        ):
            weights = softmax(scores, dim=-1)
        with annotated_range(
            ATTENTION_VALUE,
            device=device,
            record_function=record_function,
            nvtx=nvtx,
        ):
            return einsum(
                weights,
                V,
                "... query key, ... key d_v -> ... query d_v",
            )

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
    try:
        yield
    finally:
        basics_model.scaled_dot_product_attention = original


def is_annotation_name(value: Any) -> bool:
    """Return whether a profiler key is one of this assignment's ranges."""

    return isinstance(value, str) and value in REQUIRED_TRACE_LABELS
