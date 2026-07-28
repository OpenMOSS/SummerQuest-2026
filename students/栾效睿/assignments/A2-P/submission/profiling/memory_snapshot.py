"""PyTorch CUDA-memory-history helpers used after benchmark warm-up."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


@dataclass
class MemorySnapshot:
    """Own one memory-history capture window and its snapshot destination."""

    path: Path | None
    max_entries: int = 1_000_000
    enabled: bool = False

    def start(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=self.max_entries)
        self.enabled = True

    def stop_and_dump(self) -> None:
        if not self.enabled:
            return
        try:
            assert self.path is not None
            torch.cuda.memory._dump_snapshot(str(self.path))
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)
            self.enabled = False


def memory_statistics(device: torch.device) -> dict[str, int]:
    """Return active, allocated, reserved, and peak bytes with explicit names."""

    stats = torch.cuda.memory_stats(device)
    return {
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "peak_active_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def memory_metadata(snapshot_path: Path | None, statistics: dict[str, int] | None) -> dict[str, Any] | None:
    if snapshot_path is None and statistics is None:
        return None
    return {
        "snapshot_file": snapshot_path.name if snapshot_path is not None else None,
        "statistics_bytes": statistics,
    }
