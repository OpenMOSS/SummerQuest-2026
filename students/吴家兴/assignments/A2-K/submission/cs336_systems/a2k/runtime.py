"""CUDA resource, timing, and public-metadata helpers for A2-K."""

from __future__ import annotations

import math
import platform
import random
import re
import subprocess
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import torch


MIB = 2**20
GIB = 2**30
ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
MINIMUM_FREE_MIB = 22 * 1024
_OOM_ALLOCATION = re.compile(
    r"Tried to allocate ([0-9.]+) (KiB|MiB|GiB)",
    re.IGNORECASE,
)
_ABSOLUTE_PATH = re.compile(r"(?:/[A-Za-z0-9_.@+,:=-]+){2,}")


@dataclass(frozen=True)
class AllocatorConfig:
    """The process-local PyTorch allocator budget."""

    device: int
    total_memory_mib: float
    allocator_limit_mib: int
    allocator_fraction: float

    def as_public_dict(self) -> dict[str, int | float]:
        return asdict(self)


def configure_allocator(
    device: int = 0,
    allocator_limit_mib: int = ALLOCATOR_LIMIT_MIB,
) -> AllocatorConfig:
    """Set the required allocator fraction before any CUDA allocation."""

    if not torch.cuda.is_available():
        raise RuntimeError("A2-K formal runs require a CUDA device")
    if (
        torch.cuda.memory_allocated(device) != 0
        or torch.cuda.memory_reserved(device) != 0
    ):
        raise RuntimeError(
            "allocator guard must run before the first CUDA allocation"
        )
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    limit_bytes = allocator_limit_mib * MIB
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return AllocatorConfig(
        device=device,
        total_memory_mib=total_bytes / MIB,
        allocator_limit_mib=allocator_limit_mib,
        allocator_fraction=fraction,
    )


def seed_everything(seed: int) -> None:
    """Seed CPU and CUDA random-number generators."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def synchronize(device: int = 0) -> None:
    torch.cuda.synchronize(device)


def free_memory_mib(device: int = 0) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    return free_bytes / MIB


def require_formal_free_memory(
    device: int = 0,
    minimum_free_mib: int = MINIMUM_FREE_MIB,
) -> float:
    free_mib = free_memory_mib(device)
    if free_mib < minimum_free_mib:
        raise RuntimeError(
            f"formal run requires at least {minimum_free_mib} MiB free"
        )
    return free_mib


def reset_peak_memory(device: int = 0) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: int = 0) -> dict[str, float]:
    return {
        "peak_allocated_mib": (
            torch.cuda.max_memory_allocated(device) / MIB
        ),
        "peak_reserved_mib": (
            torch.cuda.max_memory_reserved(device) / MIB
        ),
    }


def _interpolated_quantile(
    sorted_values: Sequence[float],
    probability: float,
) -> float:
    position = probability * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return float(sorted_values[lower])
    weight = position - lower
    return float(
        sorted_values[lower] * (1.0 - weight)
        + sorted_values[upper] * weight
    )


def timing_summary(samples_ms: Sequence[float]) -> dict[str, Any]:
    """Summarize raw CUDA-event samples with linear quantiles."""

    if not samples_ms:
        raise ValueError("timing requires at least one sample")
    samples = [float(value) for value in samples_ms]
    ordered = sorted(samples)
    return {
        "p20_ms": _interpolated_quantile(ordered, 0.2),
        "p50_ms": _interpolated_quantile(ordered, 0.5),
        "p80_ms": _interpolated_quantile(ordered, 0.8),
        "sample_count": len(samples),
        "measurement_elapsed_ms": sum(samples),
    }


def benchmark_cuda(
    measured: Callable[[], Any],
    *,
    prepare: Callable[[], Any] | None = None,
    warmup_ms: float = 100.0,
    rep_ms: float = 300.0,
    device: int = 0,
) -> dict[str, Any]:
    """Benchmark one CUDA interval with millisecond-based warm-up and reps.

    ``prepare`` runs on the same stream before the start event.  This permits
    backward-only timing while still recreating a fresh autograd graph for
    every sample.
    """

    if warmup_ms < 0 or rep_ms <= 0:
        raise ValueError("warmup_ms must be non-negative and rep_ms positive")
    prepare_call = prepare if prepare is not None else lambda: None

    warmup_elapsed = 0.0
    warmup_iterations = 0
    while warmup_iterations == 0 or warmup_elapsed < warmup_ms:
        prepare_call()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        measured()
        end.record()
        end.synchronize()
        warmup_elapsed += max(start.elapsed_time(end), 1e-6)
        warmup_iterations += 1

    reset_peak_memory(device)
    samples: list[float] = []
    measured_elapsed = 0.0
    while not samples or measured_elapsed < rep_ms:
        prepare_call()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        measured()
        end.record()
        end.synchronize()
        elapsed = float(start.elapsed_time(end))
        samples.append(elapsed)
        measured_elapsed += max(elapsed, 1e-6)

    result = timing_summary(samples)
    result.update(
        {
            "warmup_ms": warmup_ms,
            "rep_ms": rep_ms,
            "warmup_iterations": warmup_iterations,
            **peak_memory_mib(device),
        }
    )
    return result


def timed_cold_start(
    function: Callable[[], Any],
    device: int = 0,
) -> float:
    """Measure one synchronized cold invocation in wall-clock milliseconds."""

    synchronize(device)
    started = time.perf_counter_ns()
    function()
    synchronize(device)
    return (time.perf_counter_ns() - started) / 1e6


def classify_exception(error: BaseException) -> dict[str, str]:
    """Return a short public-safe status and error summary."""

    text = str(error)
    is_oom = isinstance(error, torch.OutOfMemoryError) or (
        "out of memory" in text.lower()
    )
    if is_oom:
        allocation = _OOM_ALLOCATION.search(text)
        suffix = (
            f"; attempted {allocation.group(1)} {allocation.group(2)}"
            if allocation
            else ""
        )
        return {
            "status": "oom",
            "error_type": type(error).__name__,
            "error": f"CUDA allocator out of memory{suffix}",
        }
    first_line = text.splitlines()[0] if text else type(error).__name__
    first_line = _ABSOLUTE_PATH.sub("<path>", first_line)
    return {
        "status": "error",
        "error_type": type(error).__name__,
        "error": first_line[:240],
    }


def nvidia_smi_metadata() -> dict[str, Any]:
    """Collect only the public hardware fields required by the handout."""

    fields = (
        "name",
        "memory.total",
        "memory.free",
        "driver_version",
        "power.limit",
        "pstate",
    )
    command = [
        "nvidia-smi",
        f"--query-gpu={','.join(fields)}",
        "--format=csv,noheader,nounits",
        "--id=0",
    ]
    try:
        output = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        ).stdout.strip().splitlines()[0]
    except (
        FileNotFoundError,
        subprocess.CalledProcessError,
        subprocess.TimeoutExpired,
        IndexError,
    ):
        return {}
    values = [value.strip() for value in output.split(",")]
    return {
        "gpu_name": values[0],
        "memory_total_mib": float(values[1]),
        "memory_free_mib": float(values[2]),
        "driver_version": values[3],
        "power_limit_w": float(values[4]),
        "pstate": values[5],
    }


def software_metadata() -> dict[str, Any]:
    try:
        import triton

        triton_version: str | None = triton.__version__
    except ImportError:
        triton_version = None
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_compiled_cuda": torch.version.cuda,
        "triton": triton_version,
    }
