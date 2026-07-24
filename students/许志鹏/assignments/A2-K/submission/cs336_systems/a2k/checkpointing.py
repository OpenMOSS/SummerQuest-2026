from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


def _run_layers(layers: Sequence[nn.Module], hidden_states: torch.Tensor) -> torch.Tensor:
    for layer in layers:
        hidden_states = layer(hidden_states)
    return hidden_states


class CheckpointedTransformerLM(nn.Module):
    """Run non-overlapping groups of a BasicsTransformerLM under checkpointing.

    The wrapped model is kept intact, so parameters, optimizer state, and state-dict
    semantics remain those of the original model. A ``block_size`` of ``None`` runs
    the ordinary baseline. Formal A2-K measurements use non-nested checkpointing.
    """

    def __init__(
        self,
        model: nn.Module,
        checkpoint_block_size: int | None,
        *,
        use_reentrant: bool = False,
        preserve_rng_state: bool = True,
    ) -> None:
        super().__init__()
        if checkpoint_block_size is not None and checkpoint_block_size <= 0:
            raise ValueError("checkpoint_block_size must be positive or None")
        for attribute in ("token_embeddings", "layers", "ln_final", "lm_head"):
            if not hasattr(model, attribute):
                raise TypeError(f"wrapped model is missing required attribute {attribute!r}")
        self.model = model
        self.checkpoint_block_size = checkpoint_block_size
        self.use_reentrant = use_reentrant
        self.preserve_rng_state = preserve_rng_state

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden_states = self.model.token_embeddings(token_ids)
        layers = self.model.layers
        block_size = self.checkpoint_block_size

        if block_size is None:
            hidden_states = _run_layers(layers, hidden_states)
        else:
            for start in range(0, len(layers), block_size):
                layer_group = layers[start : start + block_size]

                def run_group(inputs: torch.Tensor, group: Sequence[nn.Module] = layer_group) -> torch.Tensor:
                    return _run_layers(group, inputs)

                hidden_states = checkpoint(
                    run_group,
                    hidden_states,
                    use_reentrant=self.use_reentrant,
                    preserve_rng_state=self.preserve_rng_state,
                )

        hidden_states = self.model.ln_final(hidden_states)
        return self.model.lm_head(hidden_states)
