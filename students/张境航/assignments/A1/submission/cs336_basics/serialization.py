from typing import IO, BinaryIO
import os

import torch
from torch.nn import Module
from torch.optim import Optimizer


def save_checkpoint(
    model: Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }

    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: Module,
    optimizer: Optimizer,
) -> int:
    checkpoint = torch.load(
        src,
        map_location="cpu",
        weights_only=False,
    )

    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    return checkpoint["iteration"]