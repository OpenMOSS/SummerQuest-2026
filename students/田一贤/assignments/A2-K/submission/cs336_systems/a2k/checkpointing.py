"""Activation-checkpointing helpers used by the benchmark scripts."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch.utils.checkpoint import checkpoint


def checkpoint_blocks(
    blocks: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
    block_size: int = 1,
    *,
    use_reentrant: bool = False,
) -> torch.Tensor:
    """Run a sequence of blocks with non-overlapping checkpoint boundaries.

    Only the input at each boundary is retained.  A block size of one checks
    every layer; a block size larger than one trades some extra saved
    activations for less recomputation.  The function is deliberately
    non-nested so its memory/compute behavior is easy to measure.
    """
    if block_size < 1:
        raise ValueError("block_size must be positive")
    if not blocks:
        return x
    for start in range(0, len(blocks), block_size):
        group = tuple(blocks[start : start + block_size])

        def run_group(
            value: torch.Tensor, group: tuple[Callable, ...] = group
        ) -> torch.Tensor:
            for fn in group:
                value = fn(value)
            return value

        x = checkpoint(run_group, x, use_reentrant=use_reentrant)
    return x


def checkpoint_sequential(
    blocks: Sequence[Callable[[torch.Tensor], torch.Tensor]],
    x: torch.Tensor,
    block_size: int = 1,
) -> torch.Tensor:
    """Compatibility alias with the assignment's terminology."""
    return checkpoint_blocks(blocks, x, block_size)


__all__ = ["checkpoint_blocks", "checkpoint_sequential"]
