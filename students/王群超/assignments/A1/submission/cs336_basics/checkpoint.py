from __future__ import annotations

import os
from typing import BinaryIO, IO

import torch
import torch.nn as nn
from torch.optim import Optimizer


def save_checkpoint(
    model: nn.Module,
    optimizer: Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存 checkpoint：model state_dict + optimizer state_dict + iteration。"""
    state = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(state, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: nn.Module,
    optimizer: Optimizer,
) -> int:
    """加载 checkpoint：恢复 model 和 optimizer state，返回 iteration。"""
    state = torch.load(src, map_location="cpu", weights_only=False)
    model.load_state_dict(state["model"])
    optimizer.load_state_dict(state["optimizer"])
    return state["iteration"]
