from __future__ import annotations

import csv
import json
import math
import os
import statistics
import subprocess
import tempfile
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch


MIB = 1024**2
GIB = 1024**3
DEFAULT_ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_GPU_LIMIT_MIB = 24 * 1024
MIN_FORMAL_FREE_MIB = 22 * 1024


@dataclass(frozen=True)
class AllocatorSettings:
    device: int
    total_memory_mib: float
    allocator_limit_mib: int
    allocator_fraction: float

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)


def configure_cuda_allocator(
    device: int = 0,
    allocator_limit_mib: int = DEFAULT_ALLOCATOR_LIMIT_MIB,
) -> AllocatorSettings:
    """Apply the A2-K allocator budget before any CUDA tensor is allocated."""
    if not torch.cuda.is_available():
        raise RuntimeError("A CUDA device is required for a formal A2-K run")
    if torch.cuda.memory_allocated(device) != 0 or torch.cuda.memory_reserved(device) != 0:
        raise RuntimeError("configure_cuda_allocator must run before the first CUDA allocation")

    total_bytes = torch.cuda.get_device_properties(device).total_memory
    limit_bytes = allocator_limit_mib * MIB
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return AllocatorSettings(
        device=device,
        total_memory_mib=total_bytes / MIB,
        allocator_limit_mib=allocator_limit_mib,
        allocator_fraction=fraction,
    )


def require_formal_free_memory(device: int = 0, minimum_free_mib: int = MIN_FORMAL_FREE_MIB) -> float:
    free_bytes, _ = torch.cuda.mem_get_info(device)
    free_mib = free_bytes / MIB
    if free_mib < minimum_free_mib:
        raise RuntimeError(f"formal run requires at least {minimum_free_mib} MiB free; found {free_mib:.1f} MiB")
    return free_mib


def synchronize(device: torch.device | str | int | None = None) -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize(device)


def reset_peak_memory(device: torch.device | str | int | None = None) -> None:
    torch.cuda.reset_peak_memory_stats(device)


def peak_memory_mib(device: torch.device | str | int | None = None) -> dict[str, float]:
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / MIB,
        "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / MIB,
    }


def quantiles_ms(samples_ms: Sequence[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("at least one timing sample is required")
    ordered = sorted(float(value) for value in samples_ms)

    def interpolated(q: float) -> float:
        position = q * (len(ordered) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return ordered[lower]
        weight = position - lower
        return ordered[lower] * (1 - weight) + ordered[upper] * weight

    return {
        "p20_ms": interpolated(0.2),
        "p50_ms": interpolated(0.5),
        "p80_ms": interpolated(0.8),
    }


def timing_summary(samples_ms: Sequence[float]) -> dict[str, float | list[float]]:
    samples = [float(value) for value in samples_ms]
    summary: dict[str, float | list[float]] = {
        "samples_ms": samples,
        "mean_ms": statistics.fmean(samples),
        "sample_std_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }
    summary.update(quantiles_ms(samples))
    return summary


def _atomic_text_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=path.parent, delete=False) as handle:
        handle.write(content)
        temporary = Path(handle.name)
    os.replace(temporary, path)


def write_json(path: str | Path, payload: Any) -> None:
    destination = Path(path)
    _atomic_text_write(destination, json.dumps(payload, indent=2, sort_keys=True, ensure_ascii=False) + "\n")


def upsert_json_record(path: str | Path, record: Mapping[str, Any], key_fields: Sequence[str]) -> None:
    destination = Path(path)
    records: list[dict[str, Any]] = []
    if destination.exists():
        loaded = json.loads(destination.read_text(encoding="utf-8"))
        if not isinstance(loaded, list):
            raise ValueError(f"{destination} must contain a JSON list")
        records = loaded

    key = tuple(record[field] for field in key_fields)
    retained = [row for row in records if tuple(row.get(field) for field in key_fields) != key]
    retained.append(dict(record))
    retained.sort(key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))
    write_json(destination, retained)


def upsert_csv_rows(
    path: str | Path,
    rows: Iterable[Mapping[str, Any]],
    key_fields: Sequence[str],
    fieldnames: Sequence[str] | None = None,
) -> None:
    destination = Path(path)
    incoming = [dict(row) for row in rows]
    existing: list[dict[str, str]] = []
    existing_fieldnames: list[str] = []
    if destination.exists():
        with destination.open(newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            existing_fieldnames = list(reader.fieldnames or [])
            existing = list(reader)

    incoming_keys = {tuple(str(row.get(field, "")) for field in key_fields) for row in incoming}
    merged: list[dict[str, Any]] = [
        row for row in existing if tuple(str(row.get(field, "")) for field in key_fields) not in incoming_keys
    ]
    merged.extend(incoming)
    merged.sort(key=lambda row: tuple(str(row.get(field, "")) for field in key_fields))

    columns = list(fieldnames or existing_fieldnames)
    if not columns:
        for row in merged:
            for column in row:
                if column not in columns:
                    columns.append(column)
    else:
        extras = {column for row in merged for column in row} - set(columns)
        columns.extend(sorted(extras))

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", newline="", dir=destination.parent, delete=False) as handle:
        writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(merged)
        temporary = Path(handle.name)
    os.replace(temporary, destination)


def _nvidia_smi_row() -> dict[str, str]:
    fields = ["name", "memory.total", "memory.free", "driver_version", "power.limit", "pstate"]
    command = ["nvidia-smi", f"--query-gpu={','.join(fields)}", "--format=csv,noheader,nounits", "--id=0"]
    try:
        output = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10).stdout.strip().splitlines()[0]
    except (FileNotFoundError, subprocess.CalledProcessError, subprocess.TimeoutExpired, IndexError):
        return {}
    values = [value.strip() for value in output.split(",")]
    return dict(zip(fields, values, strict=True))


def collect_run_metadata(
    *,
    allocator: AllocatorSettings,
    command: Sequence[str],
    seed: int,
    timer: str,
    warmup: Mapping[str, Any],
    measurement: Mapping[str, Any],
    commit: str,
    tf32_enabled: bool,
) -> dict[str, Any]:
    try:
        import triton

        triton_version: str | None = triton.__version__
    except ImportError:
        triton_version = None

    gpu = _nvidia_smi_row()
    return {
        "commit": commit,
        "seed": seed,
        "command": list(command),
        "gpu": {
            "name": gpu.get("name", torch.cuda.get_device_name(allocator.device)),
            "total_memory_mib": float(gpu.get("memory.total", allocator.total_memory_mib)),
            "free_memory_mib_at_start": float(gpu["memory.free"]) if "memory.free" in gpu else None,
            "driver_version": gpu.get("driver_version"),
            "power_limit_w": float(gpu["power.limit"]) if "power.limit" in gpu else None,
            "pstate": gpu.get("pstate"),
        },
        "software": {
            "pytorch": torch.__version__,
            "cuda": torch.version.cuda,
            "triton": triton_version,
        },
        "allocator": allocator.to_dict(),
        "tf32_enabled": tf32_enabled,
        "timer": timer,
        "warmup": dict(warmup),
        "measurement": dict(measurement),
    }
