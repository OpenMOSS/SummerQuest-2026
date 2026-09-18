"""Non-nested activation-checkpointing helpers for A2-K experiments."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint


@dataclass(frozen=True)
class CheckpointPlan:
    """Description of a non-nested checkpoint partition."""

    num_blocks: int
    checkpoint_block_size: int | None

    @property
    def is_checkpointed(self) -> bool:
        return self.checkpoint_block_size is not None

    @property
    def num_segments(self) -> int:
        if self.checkpoint_block_size is None:
            return 0
        return (self.num_blocks + self.checkpoint_block_size - 1) // self.checkpoint_block_size


def make_checkpoint_plan(num_blocks: int, checkpoint_block_size: int | None) -> CheckpointPlan:
    """Validate and construct a non-nested checkpointing plan.

    ``None`` means eager execution. A positive block size groups consecutive
    layers into independently recomputed segments, without nested calls.
    """

    if num_blocks <= 0:
        raise ValueError("num_blocks must be positive")
    if checkpoint_block_size is not None and checkpoint_block_size <= 0:
        raise ValueError("checkpoint_block_size must be positive or None")
    return CheckpointPlan(num_blocks=num_blocks, checkpoint_block_size=checkpoint_block_size)


def checkpointed_blocks(
    blocks: Sequence[nn.Module],
    hidden_states: Tensor,
    checkpoint_block_size: int | None,
    *,
    preserve_rng_state: bool = True,
) -> Tensor:
    """Run single-input Transformer blocks with optional non-nested checkpoints."""

    plan = make_checkpoint_plan(len(blocks), checkpoint_block_size)
    if not plan.is_checkpointed:
        for block in blocks:
            hidden_states = block(hidden_states)
        return hidden_states

    assert plan.checkpoint_block_size is not None
    for start in range(0, len(blocks), plan.checkpoint_block_size):
        segment = tuple(blocks[start : start + plan.checkpoint_block_size])

        def run_segment(inputs: Tensor, layers: tuple[nn.Module, ...] = segment) -> Tensor:
            output = inputs
            for layer in layers:
                output = layer(output)
            return output

        hidden_states = checkpoint(
            run_segment,
            hidden_states,
            use_reentrant=False,
            preserve_rng_state=preserve_rng_state,
        )
    return hidden_states
