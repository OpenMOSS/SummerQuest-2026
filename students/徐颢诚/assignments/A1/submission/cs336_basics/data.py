"""语言模型训练所需的数据采样工具。"""

from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 token ID 数据中随机采样一个语言模型 batch。

    对每个随机起点 ``i``，输入为 ``dataset[i : i + context_length]``，
    标签为右移一位的 ``dataset[i + 1 : i + context_length + 1]``。
    """
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    if len(dataset) <= context_length:
        raise ValueError("dataset must contain more tokens than context_length")

    # randint 的上界不包含在内，因此合法起点正好是 [0, len(dataset) - context_length)。
    start_indices = torch.randint(len(dataset) - context_length, (batch_size,))
    offsets = torch.arange(context_length)
    positions = start_indices[:, None] + offsets[None, :]

    # 只从 memmap 读取本 batch 的位置；np.array 产生可写的小副本，既避免全量复制，
    # 也避免把只读 memmap 直接交给 PyTorch 所产生的警告。
    input_ids = np.array(dataset[positions.numpy()], copy=True)
    target_ids = np.array(dataset[(positions + 1).numpy()], copy=True)
    inputs = torch.from_numpy(input_ids)
    targets = torch.from_numpy(target_ids)
    # uint16 是磁盘上常用的紧凑存储格式；模型的 embedding 索引需要 int64。
    return inputs.to(device=device, dtype=torch.long), targets.to(device=device, dtype=torch.long)
