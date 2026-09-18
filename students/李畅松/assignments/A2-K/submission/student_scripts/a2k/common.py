from __future__ import annotations

import json
import math
import statistics
import sys
import time
from pathlib import Path

import torch

DTYPES = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def parse_bool(value: str) -> bool:
    value = value.lower()
    if value in {"1", "true", "yes"}:
        return True
    if value in {"0", "false", "no"}:
        return False
    raise ValueError(f"invalid boolean: {value}")


def environment() -> dict:
    return {
        "python": sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }


def write_json(path: str, data: dict) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, indent=2) + "\n")
    print(json.dumps(data, indent=2))


def percentiles(samples: list[float]) -> dict:
    ordered = sorted(samples)
    def pick(q: float) -> float:
        pos = (len(ordered) - 1) * q
        lo, hi = math.floor(pos), math.ceil(pos)
        return ordered[lo] if lo == hi else ordered[lo] + (ordered[hi] - ordered[lo]) * (pos - lo)
    return {
        "samples": len(samples),
        "p20_ms": pick(0.2),
        "p50_ms": pick(0.5),
        "p80_ms": pick(0.8),
        "mean_ms": statistics.mean(samples),
    }


def timed_cuda(fn, warmup_ms: float, rep_ms: float) -> dict:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is required")
    torch.cuda.synchronize()
    deadline = time.perf_counter() + warmup_ms / 1000
    while time.perf_counter() < deadline:
        fn()
    torch.cuda.synchronize()
    samples = []
    deadline = time.perf_counter() + rep_ms / 1000
    while time.perf_counter() < deadline or not samples:
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return percentiles(samples)


def memory_stats() -> dict:
    return {
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
    }


def failure_record(args, exc: BaseException, stage: str) -> dict:
    result = {
        "status": "oom" if isinstance(exc, torch.OutOfMemoryError) else "error",
        "stage": stage,
        "exception_type": type(exc).__name__,
        "exception": str(exc),
        "config": vars(args),
        "environment": environment(),
    }
    if torch.cuda.is_available():
        result.update(memory_stats())
    return result
