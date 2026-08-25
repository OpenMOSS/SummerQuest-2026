from __future__ import annotations

import hashlib
import json
import os
import platform
import statistics
import subprocess
from pathlib import Path
from typing import Any

import torch

MIB = 1024**2
GIB = 1024**3
FORMAL_ALLOCATOR_LIMIT_MIB = 23 * 1024
FORMAL_HARD_LIMIT_MIB = 24 * 1024
FORMAL_MIN_FREE_MIB = 22 * 1024


def configure_formal_cuda(
    *,
    require_4090: bool = True,
    min_free_mib: int = FORMAL_MIN_FREE_MIB,
    allocator_limit_mib: int = FORMAL_ALLOCATOR_LIMIT_MIB,
) -> tuple[torch.device, dict[str, Any]]:
    """Configure the formal A2-K CUDA process before its first tensor allocation."""
    if not torch.cuda.is_available():
        raise RuntimeError("A2-K formal runs require CUDA")

    device = torch.device("cuda", 0)
    torch.cuda.set_device(device)
    properties = torch.cuda.get_device_properties(device)
    allocator_limit_bytes = allocator_limit_mib * MIB
    allocator_fraction = min(1.0, allocator_limit_bytes / properties.total_memory)
    torch.cuda.set_per_process_memory_fraction(allocator_fraction, device=device)

    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    if require_4090 and "4090" not in properties.name:
        raise RuntimeError(
            f"Formal A2-K results require an RTX 4090, found {properties.name}"
        )
    if free_bytes < min_free_mib * MIB:
        raise RuntimeError(
            f"Formal A2-K run requires at least {min_free_mib} MiB free, "
            f"found {free_bytes / MIB:.1f} MiB"
        )

    torch.set_float32_matmul_precision("high")
    metadata = {
        "gpu_name": properties.name,
        "gpu_total_memory_mib": round(total_bytes / MIB, 3),
        "gpu_free_memory_at_start_mib": round(free_bytes / MIB, 3),
        "allocator_limit_mib": allocator_limit_mib,
        "allocator_fraction": allocator_fraction,
        "hard_limit_mib": FORMAL_HARD_LIMIT_MIB,
    }
    metadata.update(_nvidia_smi_metadata())
    metadata.update(software_metadata())
    return device, metadata


def software_metadata() -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None

    return {
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "triton_version": triton_version,
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "tf32_cudnn_allowed": bool(torch.backends.cudnn.allow_tf32),
    }


def best_effort_formal_metadata() -> dict[str, Any]:
    metadata = {
        "allocator_limit_mib": FORMAL_ALLOCATOR_LIMIT_MIB,
        "hard_limit_mib": FORMAL_HARD_LIMIT_MIB,
        **software_metadata(),
    }
    if not torch.cuda.is_available():
        return metadata
    try:
        device = torch.device("cuda", 0)
        properties = torch.cuda.get_device_properties(device)
        free_bytes, total_bytes = torch.cuda.mem_get_info(device)
        metadata.update(
            {
                "gpu_name": properties.name,
                "gpu_total_memory_mib": round(total_bytes / MIB, 3),
                "gpu_free_memory_at_failure_mib": round(
                    free_bytes / MIB,
                    3,
                ),
            }
        )
        metadata.update(_nvidia_smi_metadata())
    except (RuntimeError, AssertionError):
        metadata["cuda_metadata_status"] = "unavailable"
    return metadata


def _nvidia_smi_metadata() -> dict[str, Any]:
    query = (
        "name,memory.total,memory.free,driver_version,power.limit,pstate"
    )
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
                "--id=0",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        fields = [field.strip() for field in completed.stdout.strip().split(",")]
        if len(fields) != 6:
            return {"nvidia_smi_status": "unparseable"}
        return {
            "nvidia_smi_status": "success",
            "driver_version": fields[3],
            "power_limit_w": float(fields[4]),
            "pstate": fields[5],
        }
    except (FileNotFoundError, subprocess.CalledProcessError, ValueError):
        return {"nvidia_smi_status": "unavailable"}


def latency_summary(samples_ms: list[float]) -> dict[str, Any]:
    if not samples_ms:
        return {
            "samples_ms": [],
            "p20_ms": None,
            "p50_ms": None,
            "p80_ms": None,
            "mean_ms": None,
            "sample_std_ms": None,
        }
    ordered = sorted(samples_ms)

    def quantile(q: float) -> float:
        if len(ordered) == 1:
            return ordered[0]
        position = q * (len(ordered) - 1)
        lower = int(position)
        upper = min(lower + 1, len(ordered) - 1)
        fraction = position - lower
        return ordered[lower] * (1 - fraction) + ordered[upper] * fraction

    return {
        "samples_ms": samples_ms,
        "p20_ms": quantile(0.2),
        "p50_ms": statistics.median(samples_ms),
        "p80_ms": quantile(0.8),
        "mean_ms": statistics.mean(samples_ms),
        "sample_std_ms": statistics.stdev(samples_ms)
        if len(samples_ms) > 1
        else 0.0,
    }


def peak_memory(device: torch.device) -> dict[str, float]:
    torch.cuda.synchronize(device)
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2, sort_keys=True))


def public_command(script: str, args: list[str]) -> list[str]:
    return ["python", script, *args]


def exception_payload(exc: BaseException) -> dict[str, str]:
    return {
        "error_type": type(exc).__name__,
        "error_message": str(exc),
    }


def current_commit() -> str | None:
    override = os.environ.get("A2K_SOURCE_REVISION")
    if override:
        return override
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        commit = completed.stdout.strip()
        dirty = subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=all"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        if dirty:
            return f"{commit}+dirty.{source_fingerprint()[:12]}"
        return commit
    except (FileNotFoundError, subprocess.CalledProcessError):
        return f"source-sha256:{source_fingerprint()}"


def source_fingerprint() -> str:
    repository = Path(__file__).resolve().parents[2]
    patterns = (
        "cs336_systems/a2k/*.py",
        "student_scripts/a2k/*.py",
        "tests/adapters.py",
        "cs336-basics/cs336_basics/model.py",
        "cs336-basics/cs336_basics/optimizer.py",
    )
    paths = sorted(
        {
            path
            for pattern in patterns
            for path in repository.glob(pattern)
            if path.is_file()
        }
    )
    digest = hashlib.sha256()
    for path in paths:
        digest.update(path.relative_to(repository).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
        digest.update(b"\0")
    return digest.hexdigest()
