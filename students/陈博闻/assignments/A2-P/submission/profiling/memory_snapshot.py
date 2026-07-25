from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

import torch


@contextmanager
def memory_history(snapshot_path: str | Path | None, enabled: bool, max_entries: int = 1_000_000) -> Iterator[None]:
    """Record PyTorch CUDA memory history and dump a memory_viz snapshot."""

    if not enabled:
        yield
        return
    if not torch.cuda.is_available():
        raise RuntimeError("memory profiling requires CUDA")
    if snapshot_path is None:
        raise ValueError("snapshot_path is required when memory profiling is enabled")

    path = Path(snapshot_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.cuda.memory._record_memory_history(max_entries=max_entries)
    try:
        yield
        torch.cuda.memory._dump_snapshot(str(path))
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
