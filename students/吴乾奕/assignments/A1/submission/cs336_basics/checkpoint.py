"""Model and optimizer checkpoint serialization."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch

CheckpointTarget = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """Serialize model state, optimizer state, and completed iteration."""

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": int(iteration),
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a checkpoint and return its completed iteration count."""

    checkpoint = torch.load(src, map_location="cpu")

    # The first names are what save_checkpoint writes.  Accepting the common
    # ``*_state_dict`` spelling also makes old experiment checkpoints usable.
    model_state = checkpoint.get("model", checkpoint.get("model_state_dict"))
    optimizer_state = checkpoint.get("optimizer", checkpoint.get("optimizer_state_dict"))
    if model_state is None or optimizer_state is None or "iteration" not in checkpoint:
        raise KeyError("checkpoint must contain model, optimizer, and iteration state")

    model.load_state_dict(model_state)
    optimizer.load_state_dict(optimizer_state)
    return int(checkpoint["iteration"])
