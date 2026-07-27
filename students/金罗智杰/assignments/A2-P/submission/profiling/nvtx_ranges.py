from __future__ import annotations

import math
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from einops import einsum

from cs336_basics.nn_utils import softmax


@contextmanager
def nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    if enabled:
        with torch.cuda.nvtx.range(name):
            yield
    else:
        yield


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """The basics attention implementation with profiling-only NVTX ranges."""
    use_nvtx = Q.is_cuda
    d_k = K.shape[-1]

    with nvtx_range("attention/scores", use_nvtx):
        attention_scores = einsum(Q, K, "... query d_k, ... key d_k -> ... query key") / math.sqrt(d_k)
        if mask is not None:
            attention_scores = torch.where(mask, attention_scores, float("-inf"))

    with nvtx_range("attention/softmax", use_nvtx):
        attention_weights = softmax(attention_scores, dim=-1)

    with nvtx_range("attention/value", use_nvtx):
        output = einsum(attention_weights, V, "... query key, ... key d_v -> ... query d_v")

    return output


def install_attention_nvtx() -> None:
    """Replace the module-global attention function for the current process."""
    import cs336_basics.model as model_module

    model_module.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def install_transformer_block_nvtx(model: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    """Add forward and backward NVTX ranges to every TransformerBlock."""
    handles: list[torch.utils.hooks.RemovableHandle] = []

    for index, block in enumerate(model.layers):
        forward_name = f"transformer_block/{index:02d}/forward"
        backward_name = f"transformer_block/{index:02d}/backward"

        def forward_pre_hook(_module, _inputs, name=forward_name):
            torch.cuda.nvtx.range_push(name)

        def forward_hook(_module, _inputs, _output, name=forward_name):
            del name
            torch.cuda.nvtx.range_pop()

        def backward_pre_hook(_module, _grad_output, name=backward_name):
            torch.cuda.nvtx.range_push(name)

        def backward_hook(_module, _grad_input, _grad_output, name=backward_name):
            del name
            torch.cuda.nvtx.range_pop()

        handles.append(block.register_forward_pre_hook(forward_pre_hook))
        handles.append(block.register_forward_hook(forward_hook))
        handles.append(block.register_full_backward_pre_hook(backward_pre_hook))
        handles.append(block.register_full_backward_hook(backward_hook))

    return handles
