"""Training checkpoint serialization."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch
from torch import nn


CheckpointDestination = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    if iteration < 0:
        raise ValueError("iteration must be non-negative")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: CheckpointDestination,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    map_location: str | torch.device | None = None,
) -> int:
    checkpoint = torch.load(src, map_location=map_location, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
