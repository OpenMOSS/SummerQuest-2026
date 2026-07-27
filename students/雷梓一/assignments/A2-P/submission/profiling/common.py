from __future__ import annotations

import contextlib
import json
import platform
import random
import shlex
import subprocess
import sys
from collections.abc import Iterator
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ModelConfig:
    d_model: int
    d_ff: int
    num_layers: int
    num_heads: int


MODEL_CONFIGS: dict[str, ModelConfig] = {
    # Tiny is only for CPU/local smoke tests. It is never used in reported runs.
    "tiny": ModelConfig(d_model=64, d_ff=256, num_layers=2, num_heads=4),
    "small": ModelConfig(d_model=768, d_ff=3072, num_layers=12, num_heads=12),
    "medium": ModelConfig(d_model=1024, d_ff=4096, num_layers=24, num_heads=16),
    "large": ModelConfig(d_model=1280, d_ff=5120, num_layers=36, num_heads=20),
    "xl": ModelConfig(d_model=2560, d_ff=10240, num_layers=32, num_heads=32),
    "10b": ModelConfig(d_model=4608, d_ff=12288, num_layers=50, num_heads=36),
}


def utc_now() -> str:
    return datetime.now(UTC).isoformat()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def command_string() -> str:
    """Return a reproducible command without leaking the current working directory."""
    sanitized: list[str] = []
    cwd = Path.cwd().resolve()
    for argument in sys.argv:
        path = Path(argument)
        if path.is_absolute():
            try:
                argument = str(path.resolve().relative_to(cwd))
            except ValueError:
                argument = path.name
        sanitized.append(argument)
    return shlex.join([Path(sys.executable).name, *sanitized])


def software_metadata(device: torch.device) -> dict[str, Any]:
    metadata: dict[str, Any] = {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "device_type": device.type,
    }
    if device.type == "cuda":
        props = torch.cuda.get_device_properties(device)
        metadata.update(
            {
                "gpu_name": props.name,
                "gpu_memory_bytes": props.total_memory,
                "compute_capability": f"{props.major}.{props.minor}",
                "cudnn": torch.backends.cudnn.version(),
            }
        )
        try:
            driver = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout.splitlines()[0].strip()
            metadata["driver"] = driver
        except (FileNotFoundError, subprocess.CalledProcessError, IndexError):
            metadata["driver"] = None
    return metadata


def model_config_dict(name: str) -> dict[str, int]:
    return asdict(MODEL_CONFIGS[name])


def write_json(path: str | Path, payload: Any) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(output)
    return output


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "fp32":
        return contextlib.nullcontext()
    if dtype_name != "bf16":
        raise ValueError(f"Unsupported dtype: {dtype_name}")
    if device.type != "cuda":
        return contextlib.nullcontext()
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


@contextlib.contextmanager
def range_context(name: str, device: torch.device) -> Iterator[None]:
    """Add both framework-level and CUDA/NVTX ranges when available."""
    with contextlib.ExitStack() as stack:
        stack.enter_context(torch.profiler.record_function(name))
        if device.type == "cuda":
            stack.enter_context(torch.cuda.nvtx.range(name))
        yield


def cuda_memory_metrics(device: torch.device) -> dict[str, int | None]:
    if device.type != "cuda":
        return {
            "active_bytes": None,
            "peak_active_bytes": None,
            "allocated_bytes": None,
            "peak_allocated_bytes": None,
            "reserved_bytes": None,
            "peak_reserved_bytes": None,
        }
    stats = torch.cuda.memory_stats(device)
    return {
        "active_bytes": int(stats.get("active_bytes.all.current", 0)),
        "peak_active_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "allocated_bytes": int(torch.cuda.memory_allocated(device)),
        "peak_allocated_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_bytes": int(torch.cuda.memory_reserved(device)),
        "peak_reserved_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def local_artifact_name(path: str | Path) -> str:
    """Only expose a basename in metadata intended for public submission."""
    return Path(path).name
