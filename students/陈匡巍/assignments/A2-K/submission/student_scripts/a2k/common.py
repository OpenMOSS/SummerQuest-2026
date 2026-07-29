"""Shared, public-safe utilities for A2-K experiments."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
MIB = 1024**2

MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def configure_single_gpu() -> dict[str, Any]:
    """Apply the 23 GiB allocator guard before any CUDA tensor is allocated."""
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for the formal A2-K experiments")
    properties = torch.cuda.get_device_properties(0)
    limit_bytes = ALLOCATOR_LIMIT_MIB * MIB
    fraction = min(1.0, limit_bytes / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return {
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": fraction,
        "device_name": properties.name,
        "torch_total_memory_mib": properties.total_memory / MIB,
    }


def public_gpu_metadata() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    physical_index = visible if visible.isdigit() else "0"
    query = "name,memory.total,memory.free,driver_version,power.limit,pstate"
    result = subprocess.run(
        [
            "nvidia-smi",
            f"--query-gpu={query}",
            "--format=csv,noheader,nounits",
            f"--id={physical_index}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [item.strip() for item in result.stdout.strip().split(",")]
    return {
        "gpu_name": values[0],
        "memory_total_mib": float(values[1]),
        "memory_free_mib_at_start": float(values[2]),
        "driver_version": values[3],
        "power_limit_w": float(values[4]),
        "pstate": values[5],
    }


def peak_memory() -> tuple[float, float]:
    return (
        torch.cuda.max_memory_allocated() / MIB,
        torch.cuda.max_memory_reserved() / MIB,
    )


def synchronize() -> None:
    torch.cuda.synchronize()


def percentile(values: list[float], quantile: float) -> float:
    ordered = sorted(values)
    if len(ordered) == 1:
        return ordered[0]
    position = quantile * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def timing_summary(samples_ms: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples_ms),
        "std_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
        "p20_ms": percentile(samples_ms, 0.2),
        "p50_ms": percentile(samples_ms, 0.5),
        "p80_ms": percentile(samples_ms, 0.8),
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))
