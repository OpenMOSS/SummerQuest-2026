"""Shared helpers for A2-K experiment scripts.

Every formal experiment script must call ``set_allocator_limit()`` BEFORE the
first CUDA allocation, per assignment section 3.1 (23 GiB = 23552 MiB budget).
"""

from __future__ import annotations

import json
import os
import platform
import subprocess

import torch

ALLOCATOR_LIMIT_BYTES = 23 * 1024**3
ALLOCATOR_LIMIT_MIB = 23552

MODEL_SIZES = {
    # name: (d_model, d_ff, num_layers, num_heads)
    "small": (768, 3072, 12, 12),
    "medium": (1024, 4096, 24, 16),
    "large": (1280, 5120, 36, 20),
    "xl": (2560, 10240, 32, 32),
}
VOCAB_SIZE = 10000


def set_allocator_limit(device: int = 0) -> float:
    """Set the 23 GiB allocator cap. Must run before any CUDA allocation."""
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_BYTES / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return fraction


def gpu_info(device: int = 0) -> dict:
    """Sanitized GPU metadata (no hostname / username / UUID / internal paths)."""
    out = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
            "--format=csv,noheader",
            "-i",
            str(device),
        ],
        text=True,
    ).strip()
    name, mem_total, mem_free, driver, power, pstate = [x.strip() for x in out.split(",")]
    return {
        "gpu_name": name,
        "gpu_memory_total_mib": int(mem_total.replace(" MiB", "")),
        "gpu_memory_free_at_start_mib": int(mem_free.replace(" MiB", "")),
        "driver_version": driver,
        "power_limit_w": float(power.replace(" W", "")),
        "pstate": pstate,
    }


def collect_metadata(extra: dict | None = None) -> dict:
    import triton

    md = {
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "tf32_cudnn": torch.backends.cudnn.allow_tf32,
        "allocator_fraction": min(
            1.0, ALLOCATOR_LIMIT_BYTES / torch.cuda.get_device_properties(0).total_memory
        ),
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "timer": "triton.testing.do_bench (CUDA events) / torch.cuda.Event for step timing",
        "python_version": platform.python_version(),
    }
    md.update(gpu_info(0))
    if extra:
        md.update(extra)
    return md


def git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            cwd=os.path.dirname(os.path.abspath(__file__)),
        ).strip()
    except Exception:
        # fall back to repo root (two levels up from student_scripts/a2k)
        try:
            return subprocess.check_output(
                ["git", "rev-parse", "HEAD"], text=True,
                cwd=os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."),
            ).strip()
        except Exception:
            return "unknown"


def write_json(obj, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def mib(nbytes: int | float) -> float:
    return round(nbytes / 1024**2, 2)


def peak_mem() -> dict:
    return {
        "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
        "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
    }
