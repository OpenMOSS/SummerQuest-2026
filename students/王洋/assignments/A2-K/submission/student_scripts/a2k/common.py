from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
import triton

ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024


def configure_formal_process(device: int = 0) -> float:
    if not torch.cuda.is_available():
        raise RuntimeError("A2-K formal experiments require a CUDA GPU")
    total_bytes = torch.cuda.get_device_properties(device).total_memory
    allocator_limit_bytes = ALLOCATOR_LIMIT_MIB * 2**20
    fraction = min(1.0, allocator_limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    return fraction


def command_string() -> str:
    executable = Path(sys.argv[0])
    try:
        entrypoint = executable.resolve().relative_to(Path.cwd().resolve()).as_posix()
    except ValueError:
        entrypoint = executable.name
    return "python " + " ".join([entrypoint, *sys.argv[1:]])


def _nvidia_smi(query: str) -> str | None:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")
    physical_index = visible_devices[0].strip()
    if not physical_index.isdigit():
        physical_index = "0"
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "-i",
                physical_index,
                f"--query-gpu={query}",
                "--format=csv,noheader,nounits",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip().splitlines()[0]


def public_environment(allocator_fraction: float) -> dict[str, Any]:
    properties = torch.cuda.get_device_properties(0)
    return {
        "gpu_name": torch.cuda.get_device_name(0),
        "gpu_total_memory_mib": round(properties.total_memory / 2**20, 1),
        "gpu_free_before_mib": _nvidia_smi("memory.free"),
        "driver_version": _nvidia_smi("driver_version"),
        "power_limit_w": _nvidia_smi("power.limit"),
        "pstate": _nvidia_smi("pstate"),
        "cuda_runtime": torch.version.cuda,
        "torch_version": torch.__version__,
        "triton_version": triton.__version__,
        "python_version": sys.version.split()[0],
        "tf32_matmul_allowed": bool(torch.backends.cuda.matmul.allow_tf32),
        "allocator_fraction": allocator_fraction,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
    }


def write_json(path: str | Path, payload: Any) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    os.replace(temporary, output)


def append_csv(path: str | Path, row: dict[str, Any], fieldnames: list[str]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def reset_peaks() -> None:
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats(0)


def memory_peaks() -> tuple[float, float]:
    return torch.cuda.max_memory_allocated(0) / 2**20, torch.cuda.max_memory_reserved(0) / 2**20


def benchmark_quantiles(function, *, warmup_ms: int = 100, rep_ms: int = 300) -> tuple[float, float, float]:
    values = triton.testing.do_bench(function, warmup=warmup_ms, rep=rep_ms, quantiles=[0.2, 0.5, 0.8])
    return tuple(float(value) for value in values)


def max_errors(actual: torch.Tensor, expected: torch.Tensor) -> tuple[float, float]:
    actual_fp32 = actual.float()
    expected_fp32 = expected.float()
    absolute = (actual_fp32 - expected_fp32).abs()
    relative = absolute / expected_fp32.abs().clamp_min(1e-6)
    return float(absolute.max()), float(relative.max())


def median(values: list[float]) -> float:
    return float(statistics.median(values))


def allocator_evidence(rows: list[dict[str, Any]], allocator_fraction: float) -> dict[str, Any]:
    allocated = max((float(row.get("peak_allocated_mib") or 0.0) for row in rows), default=0.0)
    reserved = max((float(row.get("peak_reserved_mib") or 0.0) for row in rows), default=0.0)
    return {
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": allocated,
        "pytorch_peak_reserved_mib": reserved,
        "within_24gib": reserved <= ALLOCATOR_LIMIT_MIB and allocated <= HARD_LIMIT_MIB,
    }


def speedup(reference_ms: float | str | None, candidate_ms: float | str | None) -> float | None:
    try:
        reference = float(reference_ms)
        candidate = float(candidate_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reference) or not math.isfinite(candidate) or candidate <= 0:
        return None
    return reference / candidate
