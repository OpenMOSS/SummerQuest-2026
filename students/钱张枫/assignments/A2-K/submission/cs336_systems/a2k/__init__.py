"""A2-K single-GPU memory and attention-kernel implementations.

The package deliberately keeps the explicit PyTorch baseline, tiled reference
implementation, and Triton implementation separate.  This makes benchmark
comparisons auditable and avoids accidentally routing a baseline through a
fused PyTorch attention operator.
"""

from .attention import explicit_attention, explicit_attention_with_lse, tiled_attention_forward
from .checkpointing import checkpointed_blocks
from .flash_pytorch import FlashAttentionPyTorchFunction
from .flash_triton import FlashAttentionTritonFunction

__all__ = [
    "FlashAttentionPyTorchFunction",
    "FlashAttentionTritonFunction",
    "checkpointed_blocks",
    "explicit_attention",
    "explicit_attention_with_lse",
    "tiled_attention_forward",
]
