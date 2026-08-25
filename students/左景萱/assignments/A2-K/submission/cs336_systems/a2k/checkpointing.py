from __future__ import annotations

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint


class CheckpointedTransformerLM(nn.Module):
    """Checkpoint consecutive, non-nested groups of Transformer blocks.

    ``block_size`` is the number of consecutive blocks recomputed by one
    checkpoint call. A value of zero leaves the wrapped model unchanged.
    """

    def __init__(self, model: nn.Module, block_size: int):
        super().__init__()
        if block_size < 0:
            raise ValueError("block_size must be non-negative")
        self.model = model
        self.block_size = block_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        if self.block_size == 0 or not torch.is_grad_enabled():
            return self.model(token_ids)

        hidden = self.model.token_embeddings(token_ids)
        layers = self.model.layers
        for start in range(0, len(layers), self.block_size):
            group = tuple(layers[start : start + self.block_size])

            def run_group(value: torch.Tensor, modules: tuple[nn.Module, ...] = group) -> torch.Tensor:
                for layer in modules:
                    value = layer(value)
                return value

            hidden = checkpoint(run_group, hidden, use_reentrant=False)
        hidden = self.model.ln_final(hidden)
        return self.model.lm_head(hidden)


__all__ = ["CheckpointedTransformerLM"]
