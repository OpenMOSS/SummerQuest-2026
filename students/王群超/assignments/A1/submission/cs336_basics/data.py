from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    从 1D numpy token ID 序列中随机采样 batch。
    x = dataset[s : s + T]
    y = dataset[s + 1 : s + T + 1]
    返回 (x, y)，形状均为 (batch_size, context_length)，dtype=torch.long。
    """
    n = len(dataset)
    max_start = n - context_length - 1
    if max_start < 0:
        raise ValueError(
            f"Dataset too short ({n}) for context_length={context_length}"
        )
    starts = np.random.randint(0, max_start + 1, size=batch_size)
    x = np.stack([dataset[s : s + context_length] for s in starts])
    y = np.stack([dataset[s + 1 : s + 1 + context_length] for s in starts])
    x_t = torch.from_numpy(x).to(device=device, dtype=torch.long)
    y_t = torch.from_numpy(y).to(device=device, dtype=torch.long)
    return x_t, y_t
