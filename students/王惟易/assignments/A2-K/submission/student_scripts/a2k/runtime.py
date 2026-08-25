from __future__ import annotations

import platform
import subprocess
from typing import Any

import torch
import triton


MIB = 1024**2
ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
MINIMUM_FREE_MIB = 22 * 1024


def _query_nvidia_smi() -> dict[str, Any]:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--id=0",
            "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
            "--format=csv,noheader,nounits",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        raise RuntimeError(f"expected one nvidia-smi row for GPU 0, got {len(lines)}")

    fields = [field.strip() for field in lines[0].split(",")]
    if len(fields) != 6:
        raise RuntimeError(f"expected six nvidia-smi fields, got {len(fields)}")

    name, total_mib, free_mib, driver_version, power_limit_w, pstate = fields
    return {
        "gpu": name,
        "gpu_memory_total_mib": float(total_mib),
        "gpu_memory_free_mib_at_start": float(free_mib),
        "driver_version": driver_version,
        "power_limit_w": float(power_limit_w),
        "pstate": pstate,
    }


def configure_single_gpu_allocator(
    minimum_free_mib: float = MINIMUM_FREE_MIB,
) -> tuple[dict[str, Any], dict[str, Any]]:
    hardware = _query_nvidia_smi()
    if hardware["gpu_memory_free_mib_at_start"] < minimum_free_mib:
        raise RuntimeError(f"insufficient free GPU memory: {hardware['gpu_memory_free_mib_at_start']:.0f} MiB available, {minimum_free_mib:.0f} MiB required")

    visible_devices = torch.cuda.device_count()
    if visible_devices != 1:
        raise RuntimeError(f"formal A2-K runs require exactly one visible CUDA device; found {visible_devices}")

    device_index = 0
    properties = torch.cuda.get_device_properties(device_index)
    allocator_limit_bytes = ALLOCATOR_LIMIT_MIB * MIB
    allocator_fraction = min(1.0, allocator_limit_bytes / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=device_index)

    allocator = {
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": allocator_fraction,
        "hard_limit_mib": HARD_LIMIT_MIB,
    }
    environment = {
        **hardware,
        "python": platform.python_version(),
        "pytorch": str(torch.__version__),
        "triton": str(triton.__version__),
        "cuda_runtime": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "visible_cuda_devices": visible_devices,
        "device_reported_total_mib": properties.total_memory / MIB,
        "tf32_matmul_allowed": torch.backends.cuda.matmul.allow_tf32,
        "float32_matmul_precision": torch.get_float32_matmul_precision(),
    }
    return allocator, environment


def synchronize() -> None:
    torch.cuda.synchronize(device=0)


def peak_memory() -> dict[str, float | bool]:
    peak_allocated_mib = torch.cuda.max_memory_allocated(0) / MIB
    peak_reserved_mib = torch.cuda.max_memory_reserved(0) / MIB
    return {
        "peak_allocated_mib": peak_allocated_mib,
        "peak_reserved_mib": peak_reserved_mib,
        "within_allocator_limit": peak_reserved_mib <= ALLOCATOR_LIMIT_MIB,
        "within_24gib": peak_reserved_mib <= HARD_LIMIT_MIB,
    }
