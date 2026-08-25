from functools import partial

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM


def _run_segment(hidden_states: torch.Tensor, *, layers: nn.ModuleList, start: int, end: int) -> torch.Tensor:
    for layer in layers[start:end]:
        hidden_states = layer(hidden_states)
    return hidden_states


def run_checkpointed_layers(
    layers: nn.ModuleList,
    hidden_states: torch.Tensor,
    checkpoint_block_size: int | None,
) -> torch.Tensor:
    if checkpoint_block_size is None:
        return _run_segment(hidden_states, layers=layers, start=0, end=len(layers))

    for start in range(0, len(layers), checkpoint_block_size):
        end = min(start + checkpoint_block_size, len(layers))

        segment_forward = partial(_run_segment, layers=layers, start=start, end=end)

        hidden_states = checkpoint(segment_forward, hidden_states, use_reentrant=False)

    return hidden_states


def transformer_lm_forward_with_checkpointing(
    model: BasicsTransformerLM,
    token_ids: torch.Tensor,
    checkpoint_block_size: int | None,
) -> torch.Tensor:
    hidden_states = model.token_embeddings(token_ids)

    hidden_states = run_checkpointed_layers(
        model.layers,
        hidden_states,
        checkpoint_block_size,
    )

    hidden_states = model.ln_final(hidden_states)
    return model.lm_head(hidden_states)
