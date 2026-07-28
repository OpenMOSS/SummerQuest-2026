"""Non-nested activation checkpointing for the fixed Transformer model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def run_layers(
    layers: Sequence[nn.Module],
    hidden_states: torch.Tensor,
) -> torch.Tensor:
    """Run a contiguous sequence of Transformer blocks."""

    for layer in layers:
        hidden_states = layer(hidden_states)
    return hidden_states


class CheckpointedTransformerLM(nn.Module):
    """Wrap a Transformer and checkpoint non-overlapping block groups."""

    def __init__(
        self,
        model: nn.Module,
        checkpoint_block_size: int | None,
        *,
        preserve_rng_state: bool = True,
    ) -> None:
        super().__init__()
        if (
            checkpoint_block_size is not None
            and checkpoint_block_size <= 0
        ):
            raise ValueError("checkpoint_block_size must be positive or None")
        for attribute in (
            "token_embeddings",
            "layers",
            "ln_final",
            "lm_head",
        ):
            if not hasattr(model, attribute):
                raise TypeError(
                    f"wrapped model is missing required attribute {attribute!r}"
                )
        self.model = model
        self.checkpoint_block_size = checkpoint_block_size
        self.preserve_rng_state = preserve_rng_state

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.token_embeddings(token_ids)
        block_size = self.checkpoint_block_size
        layers = self.model.layers

        if block_size is None:
            hidden_states = run_layers(layers, hidden_states)
        else:
            for start in range(0, len(layers), block_size):
                group = layers[start : start + block_size]

                def run_group(
                    inputs: torch.Tensor,
                    selected_layers: Sequence[nn.Module] = group,
                ) -> torch.Tensor:
                    return run_layers(selected_layers, inputs)

                hidden_states = checkpoint(
                    run_group,
                    hidden_states,
                    use_reentrant=False,
                    preserve_rng_state=self.preserve_rng_state,
                )

        hidden_states = self.model.ln_final(hidden_states)
        return self.model.lm_head(hidden_states)
