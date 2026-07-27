"""Shared runtime helpers for reproducible A2-K measurements."""

from __future__ import annotations

import contextlib
import json
import platform
import random
import subprocess
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import torch


ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024


def set_allocator_limit(device: int = 0) -> dict[str, float]:
    """Set the required 23 GiB allocator limit before any CUDA allocation."""

    if not torch.cuda.is_available():
        return {
            "total_memory_mib": 0.0,
            "allocator_limit_mib": float(ALLOCATOR_LIMIT_MIB),
            "allocator_fraction": 1.0,
        }
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    limit_bytes = 23 * 1024**3
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return {
        "total_memory_mib": total_bytes / 2**20,
        "allocator_limit_mib": float(ALLOCATOR_LIMIT_MIB),
        "allocator_fraction": fraction,
    }


def seed_all(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


@contextlib.contextmanager
def timed_cuda() -> Iterator[None]:
    synchronize()
    start = time.perf_counter()
    yield
    synchronize()
    _ = time.perf_counter() - start


def measure_callable(
    fn,
    *,
    warmup: int = 100,
    rep: int = 300,
) -> list[float]:
    """Measure a callable with CUDA synchronization; times are milliseconds."""

    for _ in range(warmup):
        fn()
    synchronize()
    values: list[float] = []
    for _ in range(rep):
        start = time.perf_counter()
        fn()
        synchronize()
        values.append((time.perf_counter() - start) * 1000.0)
    return values


def quantiles(values: list[float]) -> tuple[float, float, float]:
    if not values:
        return float("nan"), float("nan"), float("nan")
    p20, p50, p80 = np.quantile(np.asarray(values), [0.2, 0.5, 0.8])
    return float(p20), float(p50), float(p80)


def reset_peak_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    return (
        torch.cuda.max_memory_allocated() / 2**20,
        torch.cuda.max_memory_reserved() / 2**20,
    )


def nvidia_smi_metadata() -> dict[str, Any]:
    """Return public GPU fields only; omit UUID, hostname, and process data."""

    if not torch.cuda.is_available():
        return {}
    query = "name,memory.total,memory.free,driver_version,power.limit,pstate"
    try:
        raw = subprocess.check_output(
            ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader"],
            text=True,
            timeout=10,
        ).strip()
    except (OSError, subprocess.SubprocessError):
        raw = ""
    fields = [part.strip() for part in raw.split(",", 5)]
    return {
        "gpu": torch.cuda.get_device_name(0),
        "nvidia_smi": {
            "name": fields[0] if len(fields) > 0 else torch.cuda.get_device_name(0),
            "memory_total": fields[1] if len(fields) > 1 else None,
            "memory_free_at_start": fields[2] if len(fields) > 2 else None,
            "driver_version": fields[3] if len(fields) > 3 else None,
            "power_limit": fields[4] if len(fields) > 4 else None,
            "pstate": fields[5] if len(fields) > 5 else None,
        },
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "python": platform.python_version(),
        "triton": _triton_version(),
        "tf32": {
            "matmul": torch.backends.cuda.matmul.allow_tf32,
            "cudnn": torch.backends.cudnn.allow_tf32,
        },
    }


def _triton_version() -> str | None:
    try:
        import triton

        return str(triton.__version__)
    except ImportError:
        return None


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def is_oom(exc: BaseException) -> bool:
    return isinstance(exc, torch.cuda.OutOfMemoryError) or "out of memory" in str(exc).lower()


__all__ = [
    "ALLOCATOR_LIMIT_MIB",
    "HARD_LIMIT_MIB",
    "is_oom",
    "measure_callable",
    "nvidia_smi_metadata",
    "peak_memory_mib",
    "quantiles",
    "reset_peak_memory",
    "seed_all",
    "set_allocator_limit",
    "synchronize",
    "write_json",
]
