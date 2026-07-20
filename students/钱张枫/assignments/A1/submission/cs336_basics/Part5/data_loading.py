from __future__ import annotations

import operator
from typing import Any

import numpy as np
import numpy.typing as npt
import torch

type TokenArray = npt.NDArray[np.integer[Any]]


def _validate_positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer, got bool.")

    try:
        normalized_value = operator.index(value)
    except TypeError as error:
        raise TypeError(f"{name} must be an integer, got {type(value).__name__}.") from error

    if normalized_value <= 0:
        raise ValueError(f"{name} must be positive, got {normalized_value}.")
    return normalized_value


def sample_batch(
    dataset: TokenArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample random next-token prediction examples from a token array.

    The input may be a regular one-dimensional NumPy array or an ``np.memmap``.
    Only the sampled windows are materialized, so the full dataset does not need
    to fit in memory.
    """

    if not isinstance(dataset, np.ndarray):
        raise TypeError(f"dataset must be a NumPy array or memmap, got {type(dataset).__name__}.")
    if dataset.ndim != 1:
        raise ValueError(f"dataset must be one-dimensional, got shape {dataset.shape}.")
    if not np.issubdtype(dataset.dtype, np.integer):
        raise TypeError(f"dataset must contain integer token IDs, got dtype {dataset.dtype}.")

    normalized_batch_size = _validate_positive_integer(batch_size, "batch_size")
    normalized_context_length = _validate_positive_integer(context_length, "context_length")
    number_of_starting_indices = dataset.shape[0] - normalized_context_length
    if number_of_starting_indices <= 0:
        raise ValueError(
            "dataset must contain at least context_length + 1 tokens; "
            f"got {dataset.shape[0]} tokens and context_length={normalized_context_length}."
        )

    if generator is not None and generator.device.type != "cpu":
        raise ValueError("generator must be a CPU torch.Generator.")

    starting_indices = torch.randint(
        0,
        number_of_starting_indices,
        (normalized_batch_size,),
        generator=generator,
        device="cpu",
    ).numpy()
    offsets = np.arange(normalized_context_length + 1)
    sampled_tokens = np.asarray(
        dataset[starting_indices[:, None] + offsets[None, :]],
        dtype=np.int64,
    )

    target_device = torch.device(device)
    token_windows = torch.from_numpy(sampled_tokens).to(device=target_device, dtype=torch.long)
    return token_windows[:, :-1], token_windows[:, 1:]


def get_batch(
    dataset: TokenArray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compatibility entry point for sampling a language-model batch."""

    return sample_batch(dataset, batch_size, context_length, device, generator)
