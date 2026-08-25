from __future__ import annotations

import csv
import importlib.metadata
import json
import os
import statistics
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Mapping, Sequence
from importlib import import_module
from pathlib import Path
from typing import Any

import torch


MIB = 1024**2
ALLOCATOR_LIMIT_BYTES = 23 * 1024**3
MIN_FREE_BYTES = 22 * 1024**3
ATTENTION_WARMUP_MS = 100
ATTENTION_REP_MS = 300
ATTENTION_QUANTILES = (0.2, 0.5, 0.8)


def load_json(path: Path) -> dict[str, Any]:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fields: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _package_version(name: str) -> str:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return "not_installed"


def _nvidia_smi() -> dict[str, Any]:
    visible_devices = [device.strip() for device in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if device.strip()]
    command = [
        "nvidia-smi",
        *(["-i", visible_devices[0]] if len(visible_devices) == 1 else []),
        "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
        "--format=csv,noheader,nounits",
    ]
    lines = subprocess.run(command, check=True, capture_output=True, text=True).stdout.strip().splitlines()
    if len(lines) != 1:
        raise RuntimeError("Expected exactly one physical GPU from nvidia-smi")
    name, total, free, driver, power, pstate = (part.strip() for part in lines[0].split(","))
    return {
        "name": name,
        "memory_total_mib_nvidia_smi": float(total),
        "memory_free_mib_nvidia_smi": float(free),
        "driver_version": driver,
        "power_limit_w": float(power),
        "pstate": pstate,
    }


def configure_cuda(
    required_gpu: str = "RTX 4090",
    allocator_limit_bytes: int = ALLOCATOR_LIMIT_BYTES,
    min_free_bytes: int = MIN_FREE_BYTES,
) -> dict[str, Any]:
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expected exactly one visible CUDA device")
    device = torch.cuda.get_device_properties(0)
    if required_gpu not in device.name:
        raise RuntimeError(f"Expected {required_gpu}, found {device.name}")

    fraction = min(1.0, allocator_limit_bytes / device.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    free_bytes, total_bytes = torch.cuda.mem_get_info(0)
    if free_bytes < min_free_bytes:
        raise RuntimeError(f"Only {free_bytes / 1024**3:.2f} GiB is free; {min_free_bytes / 1024**3:g} GiB is required")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return {
        "gpu": {
            **_nvidia_smi(),
            "name": device.name,
            "compute_capability": f"{device.major}.{device.minor}",
            "total_memory_mib": total_bytes / MIB,
            "start_free_memory_mib": free_bytes / MIB,
        },
        "allocator": {
            "limit_bytes": allocator_limit_bytes,
            "limit_mib": allocator_limit_bytes / MIB,
            "fraction": fraction,
        },
        "software": {
            "python": sys.version.split()[0],
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": _package_version("triton"),
        },
        "tf32": {
            "cuda_matmul_allow_tf32": False,
            "cudnn_allow_tf32": False,
        },
    }


def is_cuda_oom(error: Exception) -> bool:
    return isinstance(error, torch.cuda.OutOfMemoryError) or "out of memory" in str(error).lower()


def cuda_peak_mib(reserved: bool = False) -> float | str:
    try:
        value = torch.cuda.max_memory_reserved() if reserved else torch.cuda.max_memory_allocated()
        return value / MIB
    except RuntimeError:
        return ""


def seed_all(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def allocator_evidence(fraction: float | None = None) -> dict[str, int | float]:
    return {
        "allocator_fraction": float(fraction if fraction is not None else 1.0),
        "allocator_limit_mib": ALLOCATOR_LIMIT_BYTES / MIB,
        "limit_bytes": ALLOCATOR_LIMIT_BYTES,
        "limit_mib": ALLOCATOR_LIMIT_BYTES / MIB,
        "hard_gpu_limit_gib": 24,
    }


def refresh_memory_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    sections = tuple(evidence.get(name, {}) for name in ("checkpointing", "attention_baseline", "compile_comparison", "flash_benchmark"))
    allocated = [float(section["highest_peak_allocated_mib"]) for section in sections if section.get("highest_peak_allocated_mib") is not None]
    reserved = [float(section["highest_peak_reserved_mib"]) for section in sections if section.get("highest_peak_reserved_mib") is not None]
    evidence["hard_limit_mib"] = 24 * 1024
    evidence["pytorch_peak_allocated_mib"] = max(allocated, default=None)
    evidence["pytorch_peak_reserved_mib"] = max(reserved, default=None)
    evidence["within_24gib"] = bool(reserved) and max(reserved) <= evidence["hard_limit_mib"]
    return evidence


def benchmark_cuda(step: Callable[[], Any]) -> tuple[float, float, float]:
    values = import_module("triton.testing").do_bench(
        step,
        warmup=ATTENTION_WARMUP_MS,
        rep=ATTENTION_REP_MS,
        quantiles=list(ATTENTION_QUANTILES),
    )
    return float(values[0]), float(values[1]), float(values[2])


def timed_cuda_call(call: Callable[[], Any]) -> tuple[Any, float]:
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = call()
    torch.cuda.synchronize()
    return result, (time.perf_counter() - start) * 1_000


def measure_cuda_peak(step: Callable[[], Any]) -> tuple[float, float]:
    torch.cuda.synchronize()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()
    step()
    torch.cuda.synchronize()
    return torch.cuda.max_memory_allocated() / MIB, torch.cuda.max_memory_reserved() / MIB


def sample_quantiles(samples: Sequence[float]) -> tuple[float, float, float]:
    values = torch.tensor(samples, dtype=torch.float64)
    quantiles = torch.quantile(values, torch.tensor(ATTENTION_QUANTILES, dtype=torch.float64))
    return float(quantiles[0]), float(quantiles[1]), float(quantiles[2])


def latency_columns(prefix: str, values: Sequence[float]) -> dict[str, float]:
    return {f"{prefix}_ms_p{percentile}": float(value) for percentile, value in zip((20, 50, 80), values, strict=True)}


def benchmark_cuda_step(step: Callable[[], Any], warmup: int, repetitions: int) -> tuple[list[float], float, float, float]:
    for _ in range(warmup):
        step()
    torch.cuda.synchronize()

    times, max_allocated, max_reserved = [], 0.0, 0.0
    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    for _ in range(repetitions):
        torch.cuda.reset_peak_memory_stats()
        start.record()
        step()
        end.record()
        end.synchronize()
        times.append(float(start.elapsed_time(end)))
        max_allocated = max(max_allocated, torch.cuda.max_memory_allocated() / MIB)
        max_reserved = max(max_reserved, torch.cuda.max_memory_reserved() / MIB)

    return times, statistics.median(times), max_allocated, max_reserved
