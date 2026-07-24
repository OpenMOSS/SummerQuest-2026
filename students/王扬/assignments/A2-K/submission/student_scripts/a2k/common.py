from __future__ import annotations

import argparse
import csv
import json
import math
import platform
import subprocess
import time
from pathlib import Path
from typing import Any

import torch


RESULTS_DIR = Path("results")
ASSETS_DIR = Path("assets")


def ensure_dirs() -> None:
    RESULTS_DIR.mkdir(exist_ok=True)
    ASSETS_DIR.mkdir(exist_ok=True)


def set_allocator_limit() -> dict[str, float | int]:
    if not torch.cuda.is_available():
        return {"allocator_fraction": 0.0, "allocator_limit_mib": 23552}
    torch.cuda.set_device(0)
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    allocator_limit_bytes = 23 * 1024**3
    allocator_fraction = min(1.0, allocator_limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)
    torch.cuda.init()
    _prime_cublas()
    return {"allocator_fraction": allocator_fraction, "allocator_limit_mib": 23552}


def _prime_cublas() -> None:
    a = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    b = torch.empty((1, 1), device="cuda", dtype=torch.float32)
    _ = a @ b
    torch.cuda.synchronize()


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for formal A2-K benchmarks.")
    return torch.device("cuda")


def reset_peak() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak_memory_mib() -> tuple[float, float]:
    if not torch.cuda.is_available():
        return 0.0, 0.0
    torch.cuda.synchronize()
    allocated = torch.cuda.max_memory_allocated() / 1024**2
    reserved = torch.cuda.max_memory_reserved() / 1024**2
    return allocated, reserved


def quantiles_ms(samples: list[float]) -> tuple[float, float, float]:
    if not samples:
        return math.nan, math.nan, math.nan
    sorted_samples = sorted(samples)
    def pick(q: float) -> float:
        idx = min(len(sorted_samples) - 1, max(0, round(q * (len(sorted_samples) - 1))))
        return sorted_samples[idx]
    return pick(0.2), pick(0.5), pick(0.8)


def cuda_event_time_ms(fn, warmup_steps: int = 3, measurement_steps: int = 5) -> list[float]:
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
        samples.append(start.elapsed_time(end))
    return samples


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, data: Any) -> None:
    ensure_dirs()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(data, f, indent=2, sort_keys=True)
        f.write("\n")


def git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


def cuda_metadata() -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "commit": git_commit(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32 if torch.cuda.is_available() else None,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32 if torch.cuda.is_available() else None,
    }
    try:
        import triton
        metadata["triton_version"] = triton.__version__
    except Exception:
        metadata["triton_version"] = None

    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        metadata.update(
            {
                "gpu_model": props.name,
                "gpu_total_memory_mib": props.total_memory / 1024**2,
            }
        )
        try:
            smi = subprocess.check_output(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
                    "--format=csv,noheader",
                ],
                text=True,
            ).strip()
            metadata["nvidia_smi_sanitized"] = smi
        except Exception:
            metadata["nvidia_smi_sanitized"] = None
    return metadata


def add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=RESULTS_DIR)


def now_ms() -> float:
    return time.perf_counter() * 1000.0
