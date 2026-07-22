"""Checkpoint serialization helpers."""

from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch


CheckpointTarget = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    """Serialize all state required to resume training."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": int(iteration),
        },
        out,
    )


def load_checkpoint(
    src: CheckpointTarget,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore a trusted local checkpoint and return its iteration number."""
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
