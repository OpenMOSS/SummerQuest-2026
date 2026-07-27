from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path

import torch

ALLOCATOR_LIMIT_BYTES = 23 * 1024**3


def require_cuda_and_limit_allocator() -> tuple[torch.device, float]:
    if not torch.cuda.is_available():
        raise RuntimeError("A2-K formal experiments require a CUDA GPU")
    device = torch.device("cuda:0")
    total = torch.cuda.get_device_properties(device).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_BYTES / total)
    torch.cuda.set_per_process_memory_fraction(fraction, device=device)
    return device, fraction


def quantiles_ms(samples: list[float]) -> dict[str, float]:
    ordered = sorted(samples)
    def value(q: float) -> float:
        position = q * (len(ordered) - 1)
        low = int(position)
        high = min(low + 1, len(ordered) - 1)
        return ordered[low] + (ordered[high] - ordered[low]) * (position - low)
    return {"p20_ms": value(0.2), "p50_ms": value(0.5), "p80_ms": value(0.8)}


def do_bench(function) -> dict[str, float]:
    import triton
    result = triton.testing.do_bench(function, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])
    if isinstance(result, torch.Tensor):
        result = result.cpu().tolist()
    return dict(zip(("p20_ms", "p50_ms", "p80_ms"), result, strict=True))


def append_csv(path: Path, row: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def metadata(device: torch.device, fraction: float, seed: int) -> dict:
    props = torch.cuda.get_device_properties(device)
    free_bytes, total_bytes = torch.cuda.mem_get_info(device)
    query_fields = "name,memory.total,memory.free,driver_version,power.limit,pstate"
    try:
        query = subprocess.run(
            ["nvidia-smi", f"--query-gpu={query_fields}", "--format=csv,noheader,nounits"],
            check=True, capture_output=True, text=True,
        ).stdout.splitlines()[0].split(",")
        smi = [item.strip() for item in query]
    except (OSError, subprocess.CalledProcessError, IndexError):
        smi = [props.name, "unknown", "unknown", "unknown", "unknown", "unknown"]
    try:
        import triton
        triton_version = triton.__version__
    except ImportError:
        triton_version = "unavailable"
    return {
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "seed": seed,
        "command": " ".join(sys.argv),
        "gpu": props.name,
        "total_memory_mib": total_bytes / 2**20,
        "free_memory_mib_at_start": free_bytes / 2**20,
        "driver": smi[3], "power_limit_w": smi[4], "pstate": smi[5],
        "cuda_runtime": torch.version.cuda, "pytorch": torch.__version__, "triton": triton_version,
        "tf32_matmul": torch.backends.cuda.matmul.allow_tf32,
        "allocator_limit_mib": ALLOCATOR_LIMIT_BYTES / 2**20,
        "allocator_fraction": fraction,
        "timing": "triton.testing.do_bench(warmup=100ms, rep=300ms, quantiles=0.2/0.5/0.8)",
    }


def write_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def peak_memory() -> dict[str, float]:
    return {
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 2**20,
    }
