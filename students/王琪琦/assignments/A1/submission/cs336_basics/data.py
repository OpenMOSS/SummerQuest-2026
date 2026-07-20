from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample random next-token prediction windows from a 1D token array."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    num_starting_positions = len(dataset) - context_length
    if num_starting_positions <= 0:
        raise ValueError("dataset must contain more than context_length tokens")

    starts = torch.randint(0, num_starting_positions, (batch_size,))
    offsets = torch.arange(context_length + 1)
    indices = (starts[:, None] + offsets[None, :]).numpy()
    windows = torch.as_tensor(np.asarray(dataset[indices]), dtype=torch.long)
    return windows[:, :-1].to(device), windows[:, 1:].to(device)
