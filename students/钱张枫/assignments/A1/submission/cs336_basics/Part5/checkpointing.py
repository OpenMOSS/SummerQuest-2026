from __future__ import annotations

import operator
import os
from collections.abc import Mapping
from typing import IO, Any, BinaryIO, NotRequired, TypedDict, cast

import torch

type CheckpointStream = str | os.PathLike[str] | BinaryIO | IO[bytes]


class Checkpoint(TypedDict):
    model_state_dict: dict[str, Any]
    optimizer_state_dict: dict[str, Any]
    iteration: int
    training_state: NotRequired[dict[str, Any]]


def _normalize_iteration(iteration: int) -> int:
    if isinstance(iteration, bool):
        raise TypeError("iteration must be an integer, got bool.")

    try:
        normalized_iteration = operator.index(iteration)
    except TypeError as error:
        raise TypeError(f"iteration must be an integer, got {type(iteration).__name__}.") from error

    if normalized_iteration < 0:
        raise ValueError(f"iteration must be non-negative, got {normalized_iteration}.")
    return normalized_iteration


def _get_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        try:
            return next(model.buffers()).device
        except StopIteration:
            return torch.device("cpu")


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: CheckpointStream,
    *,
    training_state: Mapping[str, Any] | None = None,
) -> None:
    """Serialize model, optimizer, and training progress to a checkpoint."""

    checkpoint: Checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": _normalize_iteration(iteration),
    }
    if training_state is not None:
        checkpoint["training_state"] = dict(training_state)
    torch.save(checkpoint, out)


def load_checkpoint(
    src: CheckpointStream,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:
    """Restore model and optimizer state, returning the completed iteration count."""

    raw_checkpoint = _load_raw_checkpoint(src, model)
    iteration = _restore_checkpoint(raw_checkpoint, model, optimizer)
    return iteration


def load_training_checkpoint(
    src: CheckpointStream,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> tuple[int, dict[str, Any] | None]:
    """Restore a checkpoint and return optional training-loop metadata."""

    raw_checkpoint = _load_raw_checkpoint(src, model)
    iteration = _restore_checkpoint(raw_checkpoint, model, optimizer)
    training_state = raw_checkpoint.get("training_state")
    if training_state is None:
        return iteration, None
    if not isinstance(training_state, Mapping):
        raise ValueError("checkpoint training_state must be a mapping when provided.")
    return iteration, dict(training_state)


def _load_raw_checkpoint(src: CheckpointStream, model: torch.nn.Module) -> Mapping[str, Any]:
    raw_checkpoint = torch.load(src, map_location=_get_model_device(model), weights_only=True)
    if not isinstance(raw_checkpoint, Mapping):
        raise ValueError("checkpoint must contain a mapping.")
    return cast(Mapping[str, Any], raw_checkpoint)


def _restore_checkpoint(
    raw_checkpoint: Mapping[str, Any],
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    required_keys = {"model_state_dict", "optimizer_state_dict", "iteration"}
    missing_keys = required_keys.difference(raw_checkpoint)
    if missing_keys:
        missing_keys_text = ", ".join(sorted(missing_keys))
        raise ValueError(f"checkpoint is missing required keys: {missing_keys_text}.")

    model_state_dict = raw_checkpoint["model_state_dict"]
    optimizer_state_dict = raw_checkpoint["optimizer_state_dict"]
    if not isinstance(model_state_dict, Mapping):
        raise ValueError("checkpoint model_state_dict must be a mapping.")
    if not isinstance(optimizer_state_dict, Mapping):
        raise ValueError("checkpoint optimizer_state_dict must be a mapping.")

    iteration = _normalize_iteration(raw_checkpoint["iteration"])
    model.load_state_dict(cast(dict[str, Any], model_state_dict))
    optimizer.load_state_dict(cast(dict[str, Any], optimizer_state_dict))
    return iteration
