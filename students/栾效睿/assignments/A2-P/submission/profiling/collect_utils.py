"""Shared, non-measuring helpers for the CUDA experiment collectors."""

from __future__ import annotations

import re
import subprocess
from collections.abc import Iterable, Sequence
from pathlib import Path
from uuid import uuid4

import torch


def require_cuda() -> None:
    """Fail before touching result files when this host cannot run the experiment."""

    if not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA is unavailable. Run this collector on the target CUDA machine; no result files were changed."
        )


def command_display(command: Sequence[str]) -> str:
    """Return a readable, reproducible command without an absolute interpreter path."""

    executable = "python" if command and command[0].endswith(("python", "python3")) else command[0]
    root = Path.cwd().resolve()

    def visible(argument: str) -> str:
        path = Path(argument)
        if not path.is_absolute():
            return argument
        try:
            return str(path.resolve().relative_to(root))
        except ValueError:
            return path.name

    return " ".join(visible(argument) for argument in (executable, *command[1:]))


def failure_kind(completed: subprocess.CompletedProcess[str]) -> str:
    """Classify an expected CUDA OOM without persisting terminal output or host paths."""

    text = f"{completed.stdout or ''}\n{completed.stderr or ''}".lower()
    return "cuda_oom" if is_cuda_oom_text(text) else "subprocess_failed"


def is_cuda_oom_text(text: str) -> bool:
    """Return whether a CUDA error message denotes an out-of-memory condition."""

    normalized = text.lower()
    return "outofmemoryerror" in normalized or "out of memory" in normalized or "cuda oom" in normalized


_REQUESTED_ALLOCATION = re.compile(
    r"(?:tried\s+to\s+allocate|allocate(?:d|ing)?)\s+([0-9]+(?:\.[0-9]+)?)\s*([kmgtpe]?i?b)\b",
    flags=re.IGNORECASE,
)
_BYTE_MULTIPLIERS = {
    "b": 1,
    "kb": 1_000,
    "mb": 1_000**2,
    "gb": 1_000**3,
    "tb": 1_000**4,
    "pb": 1_000**5,
    "eb": 1_000**6,
    "kib": 1_024,
    "mib": 1_024**2,
    "gib": 1_024**3,
    "tib": 1_024**4,
    "pib": 1_024**5,
    "eib": 1_024**6,
}


def requested_allocation_bytes(error_text: str) -> int | None:
    """Extract PyTorch's requested CUDA allocation from an OOM message when present.

    The original exception is deliberately not persisted.  This helper returns only
    the numeric byte count that can be safely included in result metadata.
    """

    match = _REQUESTED_ALLOCATION.search(error_text)
    if match is None:
        return None
    try:
        value = float(match.group(1))
        result = int(value * _BYTE_MULTIPLIERS[match.group(2).lower()])
    except (OverflowError, ValueError):
        return None
    return result if result >= 0 else None


def publish_files_transactionally(pairs: Iterable[tuple[Path, Path]]) -> None:
    """Publish a validated file group and restore all old files on failure.

    Filesystem rename is atomic per file rather than for a group.  The helper
    first moves public files to sibling backups, then moves every staged file
    into place.  If any operation fails, it removes newly published files and
    restores the old group, preventing a mixed old/new result set.
    """

    pair_list = list(pairs)
    if not pair_list:
        raise ValueError("No staged artifacts were supplied for publication.")
    destinations = [destination for _, destination in pair_list]
    if len(destinations) != len(set(destinations)):
        raise ValueError("A publication cannot target the same public file twice.")

    transaction = uuid4().hex
    backups: list[tuple[Path, Path]] = []
    published: list[Path] = []
    try:
        for staged, destination in pair_list:
            if not staged.is_file():
                raise ValueError(f"Validated staged artifact is missing: {staged}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            if destination.exists() and not destination.is_file():
                raise ValueError(f"Public artifact destination is not a file: {destination}")

        for index, (_, destination) in enumerate(pair_list):
            if destination.exists():
                backup = destination.with_name(f".{destination.name}.repair-{transaction}-{index}.bak")
                destination.replace(backup)
                backups.append((destination, backup))

        for staged, destination in pair_list:
            staged.replace(destination)
            published.append(destination)
    except BaseException:
        for destination in reversed(published):
            destination.unlink(missing_ok=True)
        for destination, backup in reversed(backups):
            if destination.exists():
                destination.unlink()
            if backup.exists():
                backup.replace(destination)
        raise
    else:
        for _, backup in backups:
            backup.unlink(missing_ok=True)
