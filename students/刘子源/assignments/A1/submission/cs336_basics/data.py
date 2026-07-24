from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray, batch_size: int, context_length: int, device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    # 数据短成这样就别硬切了，后面 np.stack 报错更难看懂。
    if dataset.ndim != 1 or len(dataset) <= context_length:
        raise ValueError("dataset must be a 1-D sequence longer than context_length")

    start_where = np.random.randint(0, len(dataset) - context_length, size=batch_size)
    batch_x = np.stack([dataset[pos : pos + context_length] for pos in start_where])
    batch_y = np.stack([dataset[pos + 1 : pos + context_length + 1] for pos in start_where])
    return (
        torch.as_tensor(batch_x, dtype=torch.long, device=device),
        torch.as_tensor(batch_y, dtype=torch.long, device=device),
    )
