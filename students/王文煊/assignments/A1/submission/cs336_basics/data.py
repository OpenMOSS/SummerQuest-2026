"""Data loading and checkpoint (de)serialization."""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``batch_size`` (input, target) sequences of length ``context_length``.

    ``dataset`` is a 1D array of token ids. Targets are inputs shifted right by one.
    """
    max_start = len(dataset) - context_length
    starts = np.random.randint(0, max_start, size=batch_size)
    inputs = np.stack([dataset[s : s + context_length] for s in starts])
    targets = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])
    x = torch.from_numpy(inputs).long().to(device)
    y = torch.from_numpy(targets).long().to(device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """Serialize model, optimizer and iteration counter."""
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        },
        out,
    )


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model/optimizer state; returns the saved iteration count."""
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]
