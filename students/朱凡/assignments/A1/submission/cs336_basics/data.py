"""Random-access batches for next-token language modeling."""

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
    """Sample input/next-token pairs from an ndarray or memory map."""

    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token array")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch_size and context_length must be positive")
    number_of_starts = len(dataset) - context_length
    if number_of_starts <= 0:
        raise ValueError("dataset must contain at least context_length + 1 tokens")

    starting_indices = torch.randint(0, number_of_starts, (batch_size,)).tolist()
    inputs_np = np.stack(
        [np.array(dataset[start : start + context_length], dtype=np.int64, copy=True) for start in starting_indices]
    )
    targets_np = np.stack(
        [
            np.array(dataset[start + 1 : start + context_length + 1], dtype=np.int64, copy=True)
            for start in starting_indices
        ]
    )
    inputs = torch.from_numpy(inputs_np).to(device=device, non_blocking=True)
    targets = torch.from_numpy(targets_np).to(device=device, non_blocking=True)
    return inputs, targets
