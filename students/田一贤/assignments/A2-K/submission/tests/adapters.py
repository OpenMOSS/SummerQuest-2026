"""Official-test adapters for this student's A2-K implementation."""

from __future__ import annotations

from cs336_systems.a2k.attention import FlashAttentionPyTorch, FlashAttentionTriton


def get_flashattention_autograd_function_pytorch() -> type:
    return FlashAttentionPyTorch


def get_flashattention_autograd_function_triton() -> type:
    return FlashAttentionTriton


def get_ddp(module):
    raise NotImplementedError("A2-D is outside A2-K")


def ddp_on_after_backward(ddp_model, optimizer):
    raise NotImplementedError("A2-D is outside A2-K")


def get_fsdp(module, compute_dtype=None):
    raise NotImplementedError("A2-D is outside A2-K")


def fsdp_on_after_backward(fsdp_model, optimizer):
    raise NotImplementedError("A2-D is outside A2-K")


def fsdp_gather_full_params(fsdp_model):
    raise NotImplementedError("A2-D is outside A2-K")


def get_sharded_optimizer(params, optimizer_cls, **kwargs):
    raise NotImplementedError("A2-D is outside A2-K")
