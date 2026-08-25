from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Any, Callable

import torch


ALLOCATOR_LIMIT_MIB = 23 * 1024


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda")


def allocator_guard(device: str = "cuda") -> dict[str, Any]:
    """Set the required 23 GiB allocator cap before any tensor allocation."""
    if not torch.cuda.is_available() or device == "cpu":
        return {"allocator_limit_mib": ALLOCATOR_LIMIT_MIB, "allocator_fraction": None}
    props = torch.cuda.get_device_properties(0)
    fraction = min(1.0, (23 * 1024**3) / props.total_memory)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return {"allocator_limit_mib": ALLOCATOR_LIMIT_MIB, "allocator_fraction": fraction}


def sync(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.synchronize()


def memory_stats() -> dict[str, float]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": 0.0, "peak_reserved_mib": 0.0}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }


def reset_peak(device: str) -> None:
    if device.startswith("cuda") and torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()


def zero_grads(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        if tensor.grad is not None:
            tensor.grad = None


def timed(fn: Callable[[], Any], device: str, warmup: int = 10, rep: int = 30) -> list[float]:
    for _ in range(warmup):
        fn()
    sync(device)
    out: list[float] = []
    for _ in range(rep):
        sync(device)
        t0 = time.perf_counter()
        fn()
        sync(device)
        out.append((time.perf_counter() - t0) * 1e3)
    return out


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p20_ms": float("nan"), "p50_ms": float("nan"), "p80_ms": float("nan")}
    x = torch.tensor(values, dtype=torch.float64)
    q = torch.quantile(x, torch.tensor([0.2, 0.5, 0.8], dtype=torch.float64)).tolist()
    return {"p20_ms": q[0], "p50_ms": q[1], "p80_ms": q[2]}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def append_csv(path: Path, row: dict[str, Any], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = fieldnames or list(row)
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        if not exists:
            w.writeheader()
        w.writerow(row)


def environment_metadata(device: str = "cuda") -> dict[str, Any]:
    info: dict[str, Any] = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": bool(torch.cuda.is_available()),
        "device_requested": device,
        "tf32_matmul": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn": bool(torch.backends.cudnn.allow_tf32),
    }
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info.update({"gpu_name": p.name, "gpu_total_mib": p.total_memory / 2**20, "cuda_runtime": torch.version.cuda})
    return info


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None
