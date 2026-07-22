"""Random next-token batch sampling for one-dimensional token corpora."""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray[np.integer],
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[Tensor, Tensor]:
    """Sample input/target sequences that are offset by exactly one token.

    Indexing only the requested windows keeps this function compatible with
    ``numpy.memmap`` and arrays opened with ``mmap_mode='r'``.
    """

    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if context_length <= 0:
        raise ValueError("context_length must be positive")

    num_possible_starts = len(dataset) - context_length
    if num_possible_starts <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens; "
            f"got {len(dataset)} tokens and context_length={context_length}"
        )

    starts = np.random.randint(0, num_possible_starts, size=batch_size)
    offsets = np.arange(context_length + 1)
    sampled = np.asarray(dataset[starts[:, None] + offsets[None, :]])

    batch = torch.as_tensor(sampled, dtype=torch.long).to(device)
    return batch[:, :-1], batch[:, 1:]
