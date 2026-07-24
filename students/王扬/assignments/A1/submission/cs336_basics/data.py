from __future__ import annotations

import numpy as np
import torch


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start = len(dataset) - context_length
    if max_start <= 0:
        raise ValueError("Dataset must be longer than context_length.")

    starts = torch.randint(0, max_start, (batch_size,))
    x_np = np.stack([dataset[start : start + context_length] for start in starts.tolist()])
    y_np = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts.tolist()])

    x = torch.as_tensor(x_np, dtype=torch.long, device=device)
    y = torch.as_tensor(y_np, dtype=torch.long, device=device)
    return x, y
