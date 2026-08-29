from __future__ import annotations

from cs336_systems.a2k import FlashAttentionPyTorchFunction, FlashAttentionTritonFunction


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorchFunction


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTritonFunction
