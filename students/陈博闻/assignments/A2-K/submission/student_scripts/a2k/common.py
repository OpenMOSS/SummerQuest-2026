from __future__ import annotations

import csv
import json
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch


ALLOCATOR_LIMIT_MIB = 23 * 1024


def set_allocator_limit(device: int = 0, limit_mib: int = ALLOCATOR_LIMIT_MIB) -> dict[str, float]:
    total = torch.cuda.get_device_properties(device).total_memory
    limit_bytes = limit_mib * 1024**2
    fraction = min(1.0, limit_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return {"allocator_limit_mib": float(limit_mib), "allocator_fraction": float(fraction)}


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("A2-K formal measurements require one CUDA GPU")
    set_allocator_limit(0)
    torch.cuda.set_device(0)
    return torch.device("cuda:0")


def mib(num_bytes: int) -> float:
    return num_bytes / 1024**2


def reset_peak_memory() -> None:
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()


def peak_memory() -> dict[str, float]:
    torch.cuda.synchronize()
    return {
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
    }


def percentile(values: list[float], q: float) -> float:
    if not values:
        return float("nan")
    values = sorted(values)
    idx = min(len(values) - 1, max(0, round((len(values) - 1) * q)))
    return values[idx]


def timing_summary(samples_ms: list[float]) -> dict[str, Any]:
    return {
        "samples_ms": samples_ms,
        "p20_ms": percentile(samples_ms, 0.2),
        "p50_ms": percentile(samples_ms, 0.5),
        "p80_ms": percentile(samples_ms, 0.8),
        "mean_ms": statistics.fmean(samples_ms) if samples_ms else float("nan"),
    }


def cuda_event_bench(fn, warmup_steps: int, measurement_steps: int) -> list[float]:
    for _ in range(warmup_steps):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(measurement_steps):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        samples.append(float(start.elapsed_time(end)))
    return samples


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"no rows to write: {path}")
    fields = list(rows[0].keys())
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)


def public_gpu_metadata() -> dict[str, Any]:
    smi = None
    try:
        smi = subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip()
    except Exception:
        smi = "unavailable"
    return {
        "nvidia_smi_public_query": smi,
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
    }
