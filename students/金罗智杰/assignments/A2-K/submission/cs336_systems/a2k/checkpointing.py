"""Activation-checkpointed execution for the CS336 Transformer."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch.utils.checkpoint import checkpoint


def _run_layer_group(x: torch.Tensor, layers: Sequence[torch.nn.Module]) -> torch.Tensor:
    for layer in layers:
        x = layer(x)
    return x


def transformer_forward_with_checkpointing(
    model: torch.nn.Module,
    token_ids: torch.Tensor,
    checkpoint_block_size: int | None,
) -> torch.Tensor:
    """Run the model with non-nested checkpoints around contiguous layer groups."""
    x = model.token_embeddings(token_ids)
    layers = tuple(model.layers)

    if checkpoint_block_size is None:
        x = _run_layer_group(x, layers)
    else:
        if checkpoint_block_size <= 0:
            raise ValueError("checkpoint_block_size must be positive or None")
        for start in range(0, len(layers), checkpoint_block_size):
            layer_group = layers[start : start + checkpoint_block_size]
            x = checkpoint(
                _run_layer_group,
                x,
                layer_group,
                use_reentrant=False,
                preserve_rng_state=False,
            )

    return model.lm_head(model.ln_final(x))
