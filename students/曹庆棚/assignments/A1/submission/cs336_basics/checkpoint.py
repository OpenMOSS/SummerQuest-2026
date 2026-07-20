from __future__ import annotations

import os
import random
from typing import IO, BinaryIO

import numpy as np
import torch
from torch import nn
from torch.optim import Optimizer


CheckpointTarget = str | os.PathLike | BinaryIO | IO[bytes]


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: CheckpointTarget,
) -> None:
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": int(iteration),
            "rng_state": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        },
        out,
    )


def load_checkpoint(
    src: CheckpointTarget,
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    checkpoint = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    rng_state = checkpoint.get("rng_state")
    if rng_state is not None:
        random.setstate(rng_state["python"])
        np.random.set_state(rng_state["numpy"])
        torch.set_rng_state(rng_state["torch"])
        if rng_state["cuda"] is not None and torch.cuda.is_available():
            torch.cuda.set_rng_state_all(rng_state["cuda"])
    return int(checkpoint["iteration"])
