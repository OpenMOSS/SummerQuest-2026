"""Reusable NVTX and torch.profiler phase annotations."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import ExitStack, contextmanager

import torch
import cs336_basics.model as basics_model
from cs336_basics.nn_utils import softmax
from einops import einsum
from jaxtyping import Bool, Float
from torch import Tensor


@contextmanager
def phase(name: str, *, nvtx: bool, record_function: bool) -> Iterator[None]:
    """Mark one phase for Nsight NVTX and/or torch.profiler."""

    with ExitStack() as stack:
        if record_function:
            stack.enter_context(torch.profiler.record_function(name))
        if nvtx:
            torch.cuda.nvtx.range_push(name)
            stack.callback(torch.cuda.nvtx.range_pop)
        yield


def install_attention_ranges(*, nvtx: bool, record_function: bool) -> None:
    """Replace the basics attention helper with an equivalent, annotated version."""

    def annotated_scaled_dot_product_attention(
        Q: Float[Tensor, " ... queries d_k"],
        K: Float[Tensor, " ... keys d_k"],
        V: Float[Tensor, " ... keys d_v"],
        mask: Bool[Tensor, " ... queries keys"] | None = None,
    ) -> Float[Tensor, " ... queries d_v"]:
        with phase("attention/scores", nvtx=nvtx, record_function=record_function):
            scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / (K.shape[-1] ** 0.5)
        with phase("attention/softmax", nvtx=nvtx, record_function=record_function):
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
            weights = softmax(scores, dim=-1)
        with phase("attention/value", nvtx=nvtx, record_function=record_function):
            return einsum(weights, V, "... query key, ... key d_v -> ... query d_v")

    basics_model.scaled_dot_product_attention = annotated_scaled_dot_product_attention
