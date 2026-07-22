from __future__ import annotations

import math
import platform
import random
import statistics
import subprocess
from contextlib import nullcontext
from typing import Any

import torch


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2560, "d_ff": 10240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4608, "d_ff": 12288, "num_layers": 50, "num_heads": 36},
}


def git_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def nvidia_driver_version() -> str | None:
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        return None
    versions = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ",".join(versions) or None


def environment_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "device": str(device),
        "python_version": platform.python_version(),
        "torch_version": torch.__version__,
        "compiled_cuda": torch.version.cuda,
        "git_commit": git_commit(),
    }
    if device.type == "cuda" and torch.cuda.is_available():
        index = device.index if device.index is not None else torch.cuda.current_device()
        metadata.update(
            {
                "gpu_name": torch.cuda.get_device_name(index),
                "driver_version": nvidia_driver_version(),
                "bf16_supported": torch.cuda.is_bf16_supported(),
            }
        )
    else:
        metadata.update({"gpu_name": None, "driver_version": None, "bf16_supported": None})
    return metadata


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "fp32":
        return nullcontext()
    if dtype_name != "bf16":
        raise ValueError(f"unsupported dtype: {dtype_name}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def sample_statistics(values: list[float]) -> dict[str, float]:
    if not values:
        raise ValueError("at least one timing value is required")
    mean = statistics.mean(values)
    sample_std = statistics.stdev(values) if len(values) > 1 else 0.0
    cv = sample_std / mean if not math.isclose(mean, 0.0) else 0.0
    return {"mean_ms": mean, "sample_std_ms": sample_std, "cv": cv}
