"""Shared measurement helpers for A2-P."""

from __future__ import annotations

import csv
import json
import os
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

MIB = 1024**2
ALLOCATOR_LIMIT_MIB = 23 * 1024
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
    "10b": {
        "d_model": 4608,
        "d_ff": 12288,
        "num_layers": 50,
        "num_heads": 36,
    },
}


def configure_gpu() -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    properties = torch.cuda.get_device_properties(0)
    fraction = min(1.0, ALLOCATOR_LIMIT_MIB * MIB / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, 0)
    return {
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": fraction,
        "device_name": properties.name,
        "torch_total_memory_mib": properties.total_memory / MIB,
    }


def gpu_metadata() -> dict[str, Any]:
    visible = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",", 1)[0]
    physical_index = visible if visible.isdigit() else "0"
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
            "--format=csv,noheader,nounits",
            f"--id={physical_index}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    values = [part.strip() for part in result.stdout.strip().split(",")]
    return {
        "gpu_name": values[0],
        "memory_total_mib": float(values[1]),
        "memory_free_mib_at_start": float(values[2]),
        "driver_version": values[3],
        "power_limit_w": float(values[4]),
        "pstate": values[5],
    }


def summarize_timings(samples_ms: list[float]) -> dict[str, float]:
    mean = statistics.fmean(samples_ms)
    standard_deviation = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "mean_ms": mean,
        "std_ms": standard_deviation,
        "cv": standard_deviation / mean if mean else 0.0,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
