from __future__ import annotations

import torch

from cs336_systems.a2k.attention import FlashAttentionPytorch, FlashAttentionTriton


def get_flashattention_autograd_function_pytorch() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2.
    """
    return FlashAttentionPytorch


def get_flashattention_autograd_function_triton() -> type:
    """
    Returns a torch.autograd.Function subclass that implements FlashAttention2
    using Triton kernels.
    """
    return FlashAttentionTriton


def get_ddp(module: torch.nn.Module) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    parameter broadcasting and gradient synchronization for
    distributed data parallel training.
    """
    from cs336_systems.distributed import DistributedDataParallel
    return DistributedDataParallel(module)


def ddp_on_after_backward(ddp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    """
    ddp_model.finish_gradient_synchronization()


def get_fsdp(module: torch.nn.Module, compute_dtype: torch.dtype | None = None) -> torch.nn.Module:
    """
    Returns a torch.nn.Module container that handles
    fully-sharded data parallel training.
    """
    from cs336_systems.fsdp import FSDP
    return FSDP(module, compute_dtype=compute_dtype)


def fsdp_on_after_backward(fsdp_model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    """
    Code to run after the backward pass is completed, but before we take
    an optimizer step.
    """
    fsdp_model.sync_gradients()


def fsdp_gather_full_params(fsdp_model: torch.nn.Module) -> dict[str, torch.Tensor]:
    """
    All-gather sharded parameters from the FSDP model to reconstruct full
    parameter tensors.
    """
    return fsdp_model.gather_full_params()


def get_sharded_optimizer(params, optimizer_cls: type[torch.optim.Optimizer], **kwargs) -> torch.optim.Optimizer:
    """
    Returns a torch.optim.Optimizer that handles optimizer state sharding
    of the given optimizer_cls on the provided parameters.
    """
    from cs336_systems.optimizer import ShardedOptimizer
    return ShardedOptimizer(params, optimizer_cls, **kwargs)
