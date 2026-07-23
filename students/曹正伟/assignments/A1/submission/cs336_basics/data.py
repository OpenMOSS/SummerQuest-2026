"""Language-model data loading helpers."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random contiguous next-token-prediction examples."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be one-dimensional")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    number_of_starts = len(dataset) - context_length
    if number_of_starts <= 0:
        raise ValueError("dataset must contain more than context_length tokens")

    starts = np.random.randint(0, number_of_starts, size=batch_size)
    inputs = np.stack([dataset[start : start + context_length] for start in starts])
    targets = np.stack([dataset[start + 1 : start + context_length + 1] for start in starts])
    return (
        torch.as_tensor(inputs, dtype=torch.long, device=device),
        torch.as_tensor(targets, dtype=torch.long, device=device),
    )
