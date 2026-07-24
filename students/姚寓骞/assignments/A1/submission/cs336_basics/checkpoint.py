from __future__ import annotations

from typing import IO, BinaryIO
import os

import torch


def save_checkpoint(model, optimizer, iteration: int, out: str | os.PathLike | BinaryIO | IO[bytes]) -> None:
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}, out)


def load_checkpoint(src: str | os.PathLike | BinaryIO | IO[bytes], model, optimizer) -> int:
    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])
