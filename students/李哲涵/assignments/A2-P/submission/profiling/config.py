from __future__ import annotations

import os
import random
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "small": {
        "vocab_size": 10_000,
        "d_model": 768,
        "num_layers": 12,
        "num_heads": 12,
        "d_ff": 3072,
    },
    "medium": {
        "vocab_size": 10_000,
        "d_model": 1024,
        "num_layers": 24,
        "num_heads": 16,
        "d_ff": 4096,
    },
    "large": {
        "vocab_size": 10_000,
        "d_model": 1280,
        "num_layers": 36,
        "num_heads": 20,
        "d_ff": 5120,
    },
    "xl": {
        "vocab_size": 10_000,
        "d_model": 2560,
        "num_layers": 32,
        "num_heads": 32,
        "d_ff": 10_240,
    },
    "10b": {
        "vocab_size": 10_000,
        "d_model": 4608,
        "num_layers": 50,
        "num_heads": 36,
        "d_ff": 12_288,
    },
}

MIB = 1024**2


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    resolved = torch.device(device)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return resolved


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def model_config(model_size: str, context_length: int) -> dict[str, int]:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"unknown model size: {model_size}")
    config = dict(MODEL_CONFIGS[model_size])
    config["context_length"] = context_length
    return config


def build_model(model_size: str, context_length: int, device: torch.device) -> BasicsTransformerLM:
    return BasicsTransformerLM(**model_config(model_size, context_length)).to(device)


def build_optimizer(model: torch.nn.Module) -> AdamW:
    return AdamW(model.parameters(), lr=1e-4)


def autocast_context(device: torch.device, dtype: str):
    if dtype == "fp32":
        return nullcontext()
    if dtype == "bf16":
        if device.type not in {"cpu", "cuda"}:
            raise ValueError(f"BF16 autocast is unsupported on {device.type}")
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    raise ValueError(f"unsupported dtype: {dtype}")


def mib(value: int | float) -> float:
    return float(value) / MIB


def public_path(value: str | os.PathLike[str]) -> str:
    path = Path(value)
    try:
        return path.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except (OSError, ValueError):
        return path.name


def public_command(module: str, arguments: list[str]) -> list[str]:
    command = [Path(sys.executable).name, "-m", module]
    path_options = {"--output", "--output-root", "--trace", "--snapshot", "--summary"}
    redact_next = False
    for argument in arguments:
        if redact_next:
            command.append(public_path(argument))
            redact_next = False
            continue
        command.append(argument)
        redact_next = argument in path_options
    return command


def _driver_version() -> str | None:
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
            timeout=5,
        )
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired):
        return None
    values = sorted({line.strip() for line in result.stdout.splitlines() if line.strip()})
    return ", ".join(values) or None


def public_environment(device: torch.device) -> dict[str, Any]:
    environment: dict[str, Any] = {
        "python": f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}",
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "device_type": device.type,
    }
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        environment.update(
            {
                "gpu": properties.name,
                "gpu_total_memory_mib": round(mib(properties.total_memory), 2),
                "compute_capability": f"{properties.major}.{properties.minor}",
                "driver": _driver_version(),
            }
        )
    return environment


def sanitize_error(exc: BaseException) -> str:
    message = str(exc).splitlines()[0].strip()
    home = str(Path.home())
    cwd = str(Path.cwd())
    message = message.replace(home, "<home>").replace(cwd, "<workspace>")
    return message[:1000]


def cuda_memory_metrics(device: torch.device) -> dict[str, float | None]:
    if device.type != "cuda":
        return {
            "allocated_mib": None,
            "reserved_mib": None,
            "peak_allocated_mib": None,
            "peak_reserved_mib": None,
            "active_mib": None,
            "peak_active_mib": None,
        }
    synchronize(device)
    stats = torch.cuda.memory_stats(device)
    return {
        "allocated_mib": round(mib(torch.cuda.memory_allocated(device)), 3),
        "reserved_mib": round(mib(torch.cuda.memory_reserved(device)), 3),
        "peak_allocated_mib": round(mib(torch.cuda.max_memory_allocated(device)), 3),
        "peak_reserved_mib": round(mib(torch.cuda.max_memory_reserved(device)), 3),
        "active_mib": round(mib(stats.get("active_bytes.all.current", 0)), 3),
        "peak_active_mib": round(mib(stats.get("active_bytes.all.peak", 0)), 3),
    }

