import numpy as np
import numpy.typing as npt
import torch
from torch import Tensor


def get_batch(
    dataset: npt.NDArray,
    batch_size: int,
    context_length: int,
    device: str,
) -> tuple[Tensor, Tensor]:
    max_start = len(dataset) - context_length

    if max_start <= 0:
        raise ValueError(
            "dataset must contain more tokens than context_length"
        )

    start_indices = np.random.randint(
        low=0,
        high=max_start,
        size=batch_size,
    )

    inputs = np.stack(
        [
            dataset[start : start + context_length]
            for start in start_indices
        ]
    )

    targets = np.stack(
        [
            dataset[start + 1 : start + context_length + 1]
            for start in start_indices
        ]
    )

    x = torch.tensor(
        inputs,
        dtype=torch.long,
        device=device,
    )

    y = torch.tensor(
        targets,
        dtype=torch.long,
        device=device,
    )

    return x, y