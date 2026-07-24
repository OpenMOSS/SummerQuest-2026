"""训练 checkpoint 的保存与恢复。"""

from __future__ import annotations

import os
from typing import IO, BinaryIO

import torch

CheckpointDestination = str | os.PathLike[str] | BinaryIO | IO[bytes]


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointDestination,
) -> None:
    """将模型、优化器和已完成的训练迭代次数序列化到 ``out``。"""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointDestination,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """从 ``src`` 恢复模型和优化器状态，并返回保存时的迭代次数。"""
    checkpoint = torch.load(src, weights_only=True)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return int(checkpoint["iteration"])
