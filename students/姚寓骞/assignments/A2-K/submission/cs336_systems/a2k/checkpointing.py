"""Non-nested activation-checkpoint helpers for a sequence of blocks."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def run_checkpointed_blocks(
    blocks: Sequence[nn.Module],
    hidden: torch.Tensor,
    block_size: int | None,
    *,
    use_reentrant: bool = False,
) -> torch.Tensor:
    """Run blocks normally or checkpoint contiguous, non-overlapping groups."""
    if block_size is None:
        for layer in blocks:
            hidden = layer(hidden)
        return hidden
    if block_size <= 0:
        raise ValueError("block_size must be positive or None")

    def run_group(x: torch.Tensor, group: tuple[nn.Module, ...]) -> torch.Tensor:
        for layer in group:
            x = layer(x)
        return x

    for start in range(0, len(blocks), block_size):
        group = tuple(blocks[start : start + block_size])
        hidden = checkpoint(run_group, hidden, group, use_reentrant=use_reentrant)
    return hidden


__all__ = ["run_checkpointed_blocks"]
