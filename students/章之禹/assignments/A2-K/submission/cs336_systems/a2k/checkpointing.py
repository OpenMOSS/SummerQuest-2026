"""Activation-checkpointing helpers for the A2-K experiments."""

from __future__ import annotations

from collections.abc import Sequence
from torch.utils.checkpoint import checkpoint
from torch import Tensor, nn


def checkpoint_sequential_blocks(
    blocks: Sequence[nn.Module],
    x: Tensor,
    block_size: int | None = None,
    *,
    use_reentrant: bool = False,
) -> Tensor:
    """Apply Transformer blocks with optional non-nested activation checkpoints.

    ``block_size=None`` is the eager baseline.  A positive block size creates
    one checkpoint boundary per group; nested checkpointing is intentionally
    left to the theoretical discussion in the report.
    """

    if block_size is None or block_size <= 0:
        for block in blocks:
            x = block(x)
        return x

    def run_group(start: int, stop: int, value: Tensor) -> Tensor:
        for index in range(start, stop):
            value = blocks[index](value)
        return value

    for start in range(0, len(blocks), block_size):
        stop = min(start + block_size, len(blocks))
        x = checkpoint(
            lambda value, start=start, stop=stop: run_group(start, stop, value),
            x,
            use_reentrant=use_reentrant,
        )
    return x


def recursively_checkpoint(
    blocks: Sequence[nn.Module],
    x: Tensor,
    *,
    use_reentrant: bool = False,
) -> Tensor:
    """Theoretical fully recursive strategy: one checkpoint per binary interval."""

    if len(blocks) <= 1:
        return blocks[0](x) if blocks else x
    midpoint = len(blocks) // 2

    def left(value: Tensor) -> Tensor:
        return recursively_checkpoint(blocks[:midpoint], value, use_reentrant=use_reentrant)

    def right(value: Tensor) -> Tensor:
        return recursively_checkpoint(blocks[midpoint:], value, use_reentrant=use_reentrant)

    return checkpoint(
        right,
        checkpoint(left, x, use_reentrant=use_reentrant),
        use_reentrant=use_reentrant,
    )


__all__ = ["checkpoint_sequential_blocks", "recursively_checkpoint"]
