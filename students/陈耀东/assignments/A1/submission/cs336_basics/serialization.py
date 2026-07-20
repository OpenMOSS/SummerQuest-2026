"""CS336 Assignment 1 的 checkpoint 保存与加载脚手架。"""

from __future__ import annotations

from typing import IO, BinaryIO
import os

import numpy as np
import torch


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: str | os.PathLike | BinaryIO | IO[bytes],
) -> None:
    """保存模型、优化器和当前 iteration。

    推荐保存结构：
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
        }

    注意：
        - out 可以是路径，也可以是二进制文件对象。
        - torch.save 可以直接处理这两类输入。
    """
    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
        "torch_rng_state": torch.get_rng_state(),
        "numpy_rng_state": _serialize_numpy_rng_state(np.random.get_state()),
    }
    if torch.cuda.is_available():
        checkpoint["cuda_rng_state_all"] = torch.cuda.get_rng_state_all()
    torch.save(checkpoint, out)


def load_checkpoint(
    src: str | os.PathLike | BinaryIO | IO[bytes],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """加载 checkpoint，并返回保存时的 iteration。

    目标：
        - 恢复 model.state_dict()
        - 恢复 optimizer.state_dict()
        - 返回 iteration

    注意：
        - src 可以是路径，也可以是二进制文件对象。
        - torch.load 后用 load_state_dict 恢复。
    """
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    if "torch_rng_state" in checkpoint:
        torch.set_rng_state(checkpoint["torch_rng_state"])
    if "numpy_rng_state" in checkpoint:
        np.random.set_state(_deserialize_numpy_rng_state(checkpoint["numpy_rng_state"]))
    if "cuda_rng_state_all" in checkpoint and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(checkpoint["cuda_rng_state_all"])
    return checkpoint["iteration"]


def _serialize_numpy_rng_state(state: tuple) -> dict[str, object]:
    """把 NumPy RNG state 转成只含安全基础类型和 tensor 的 checkpoint 数据。"""
    bit_generator, keys, position, has_gauss, cached_gaussian = state
    return {
        "bit_generator": bit_generator,
        "keys": torch.from_numpy(keys.copy()),
        "position": position,
        "has_gauss": has_gauss,
        "cached_gaussian": cached_gaussian,
    }


def _deserialize_numpy_rng_state(state: dict[str, object]) -> tuple:
    """把 checkpoint 中的数据还原成 ``numpy.random.set_state`` 所需元组。"""
    keys = state["keys"]
    if not isinstance(keys, torch.Tensor):
        raise TypeError("checkpoint 中的 NumPy RNG keys 必须是 tensor")
    return (
        state["bit_generator"],
        keys.cpu().numpy().astype(np.uint32, copy=False),
        state["position"],
        state["has_gauss"],
        state["cached_gaussian"],
    )
