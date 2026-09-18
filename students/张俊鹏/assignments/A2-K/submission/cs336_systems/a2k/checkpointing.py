from __future__ import annotations

import torch
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


class CheckpointedBasicsTransformerLM(BasicsTransformerLM):
    """BasicsTransformerLM with non-nested activation checkpoint blocks."""

    def __init__(self, *args, checkpoint_block_size: int | None = None, **kwargs):
        super().__init__(*args, **kwargs)
        if checkpoint_block_size is not None and checkpoint_block_size <= 0:
            raise ValueError("checkpoint_block_size must be positive or None")
        self.checkpoint_block_size = checkpoint_block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden_states = self.token_embeddings(x)
        block_size = self.checkpoint_block_size

        if self.training and block_size is not None:
            for block_start in range(0, len(self.layers), block_size):
                layers = tuple(self.layers[block_start : block_start + block_size])

                def run_block(states, layers=layers):
                    for layer in layers:
                        states = layer(states)
                    return states

                hidden_states = checkpoint(
                    run_block,
                    hidden_states,
                    use_reentrant=False,
                )
        else:
            for layer in self.layers:
                hidden_states = layer(hidden_states)

        hidden_states = self.ln_final(hidden_states)
        return self.lm_head(hidden_states)
