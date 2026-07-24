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
    rng: np.random.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Sample random contiguous next-token prediction windows from a token stream."""
    if dataset.ndim != 1:
        raise ValueError("dataset must be a one-dimensional token array.")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if context_length <= 0:
        raise ValueError("context_length must be positive.")

    num_possible_starts = len(dataset) - context_length
    if num_possible_starts <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens."
        )

    if rng is None:
        starting_indices = np.random.randint(
            low=0,
            high=num_possible_starts,
            size=batch_size,
        )
    else:
        starting_indices = rng.integers(
            low=0,
            high=num_possible_starts,
            size=batch_size,
        )
    windows = np.stack(
        [
            dataset[start : start + context_length + 1]
            for start in starting_indices
        ],
        axis=0,
    )
    windows_tensor = torch.as_tensor(
        windows,
        dtype=torch.long,
        device=device,
    )
    return windows_tensor[:, :-1], windows_tensor[:, 1:]
