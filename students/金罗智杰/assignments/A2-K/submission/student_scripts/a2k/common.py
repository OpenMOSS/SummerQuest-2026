"""Shared environment, measurement, and serialization helpers for A2-K."""

from __future__ import annotations

import csv
import json
import statistics
import subprocess
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import triton

ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
MIN_FREE_MIB = 22 * 1024


@dataclass(frozen=True)
class CudaEnvironment:
    allocator_fraction: float
    allocator_limit_mib: int
    gpu_name: str
    memory_total_mib: float
    memory_free_mib: float
    driver_version: str
    cuda_runtime: str
    torch_version: str
    triton_version: str
    power_limit_w: str
    pstate: str
    tf32_matmul: bool
    tf32_cudnn: bool
    visible_device_count: int


def _query_nvidia_smi() -> dict[str, str]:
    query = "name,memory.total,memory.free,driver_version,power.limit,pstate"
    completed = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        check=True,
        capture_output=True,
        text=True,
    )
    first_gpu = completed.stdout.strip().splitlines()[0]
    values = [value.strip() for value in first_gpu.split(",")]
    keys = ["gpu_name", "memory_total_mib", "memory_free_mib", "driver_version", "power_limit_w", "pstate"]
    return dict(zip(keys, values, strict=True))


def configure_cuda_environment(require_rtx4090: bool = True) -> CudaEnvironment:
    """Apply the 23 GiB allocator cap before the first CUDA tensor allocation."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA GPU is required for formal A2-K measurements")

    smi = _query_nvidia_smi()
    visible_device_count = torch.cuda.device_count()
    memory_total_mib = float(smi["memory_total_mib"])
    if require_rtx4090 and visible_device_count != 1:
        raise RuntimeError(
            f"formal A2-K measurements require exactly one CUDA-visible GPU, found {visible_device_count}"
        )
    if require_rtx4090 and "RTX 4090" not in smi["gpu_name"]:
        raise RuntimeError(f"formal A2-K measurements require RTX 4090, found {smi['gpu_name']}")
    if float(smi["memory_free_mib"]) < MIN_FREE_MIB:
        raise RuntimeError(
            f"at least {MIN_FREE_MIB} MiB free GPU memory is required; "
            f"nvidia-smi reported {smi['memory_free_mib']} MiB"
        )

    total_bytes = torch.cuda.get_device_properties(0).total_memory
    allocator_limit_bytes = ALLOCATOR_LIMIT_MIB * 1024**2
    allocator_fraction = min(1.0, allocator_limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=0)

    return CudaEnvironment(
        allocator_fraction=allocator_fraction,
        allocator_limit_mib=ALLOCATOR_LIMIT_MIB,
        gpu_name=smi["gpu_name"],
        memory_total_mib=memory_total_mib,
        memory_free_mib=float(smi["memory_free_mib"]),
        driver_version=smi["driver_version"],
        cuda_runtime=str(torch.version.cuda),
        torch_version=torch.__version__,
        triton_version=triton.__version__,
        power_limit_w=smi["power_limit_w"],
        pstate=smi["pstate"],
        tf32_matmul=torch.backends.cuda.matmul.fp32_precision != "ieee",
        tf32_cudnn=torch.backends.cudnn.conv.fp32_precision != "ieee",
        visible_device_count=visible_device_count,
    )


def synchronize() -> None:
    torch.cuda.synchronize()


def timed_call(operation) -> float:
    synchronize()
    start = time.perf_counter()
    operation()
    synchronize()
    return (time.perf_counter() - start) * 1000


def percentile(samples: list[float], quantile: float) -> float:
    if not samples:
        raise ValueError("samples must not be empty")
    ordered = sorted(samples)
    index = (len(ordered) - 1) * quantile
    lower = int(index)
    upper = min(lower + 1, len(ordered) - 1)
    weight = index - lower
    return ordered[lower] * (1 - weight) + ordered[upper] * weight


def summarize_samples(samples: list[float]) -> dict[str, float]:
    return {
        "mean_ms": statistics.fmean(samples),
        "std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
        "p20_ms": percentile(samples, 0.2),
        "p50_ms": percentile(samples, 0.5),
        "p80_ms": percentile(samples, 0.8),
    }


def peak_memory() -> tuple[float, float]:
    mib = 1024**2
    return torch.cuda.max_memory_allocated() / mib, torch.cuda.max_memory_reserved() / mib


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def environment_metadata(
    environment: CudaEnvironment,
    *,
    command: str,
    seed: int,
    warmup: Any,
    measurement: Any,
) -> dict[str, Any]:
    return {
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "command": command,
        "seed": seed,
        "environment": asdict(environment),
        "measurement": {
            "timer": "synchronized wall clock or triton.testing.do_bench",
            "warmup": warmup,
            "measurement": measurement,
        },
    }
