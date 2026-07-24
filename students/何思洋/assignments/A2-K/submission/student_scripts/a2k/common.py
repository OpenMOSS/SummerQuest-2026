from __future__ import annotations

import csv
import json
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
CS336_BASICS = ROOT / "cs336-basics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CS336_BASICS) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS))

import torch


def set_allocator_limit(limit_mib: int | None) -> dict[str, float | int | None]:
    if limit_mib is None or not torch.cuda.is_available():
        return {"allocator_fraction": None, "allocator_limit_mib": limit_mib}
    total = torch.cuda.get_device_properties(0).total_memory
    limit_bytes = limit_mib * 1024**2
    fraction = min(1.0, limit_bytes / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return {"allocator_fraction": fraction, "allocator_limit_mib": limit_mib}


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def sync() -> None:
    if torch.cuda.is_available():
        torch.cuda.synchronize()


def reset_memory() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()


def memory_stats() -> dict[str, float | None]:
    if not torch.cuda.is_available():
        return {"peak_allocated_mib": None, "peak_reserved_mib": None}
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def summarize(samples: list[float]) -> dict[str, float | None]:
    if not samples:
        return {"p20_ms": None, "p50_ms": None, "p80_ms": None, "mean_ms": None}
    ordered = sorted(samples)
    return {
        "p20_ms": ordered[round(0.2 * (len(ordered) - 1))],
        "p50_ms": statistics.median(ordered),
        "p80_ms": ordered[round(0.8 * (len(ordered) - 1))],
        "mean_ms": statistics.fmean(ordered),
    }


def metadata(args: dict[str, Any] | None = None, allocator: dict[str, Any] | None = None) -> dict[str, Any]:
    clean_args = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in (args or {}).items()
    }
    payload: dict[str, Any] = {
        "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "args": clean_args,
    }
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        payload["gpu"] = {"name": props.name, "total_memory_mib": props.total_memory / 1024**2}
        try:
            smi = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,memory.total,memory.free,driver_version,power.limit,pstate",
                    "--format=csv,noheader",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            payload["nvidia_smi"] = smi.stdout.strip().splitlines()[0]
        except Exception:
            pass
    if allocator is not None:
        payload["allocator"] = allocator
    return payload


def bench(fn, warmup: int, steps: int) -> tuple[list[float], str, str]:
    status = "ok"
    error = ""
    try:
        for _ in range(warmup):
            fn()
            sync()
        reset_memory()
        samples = []
        for _ in range(steps):
            start = time.perf_counter()
            fn()
            sync()
            samples.append((time.perf_counter() - start) * 1000.0)
        return samples, status, error
    except torch.cuda.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return [], "oom", str(exc).splitlines()[0]
    except RuntimeError as exc:
        return [], "error", str(exc).splitlines()[0]
