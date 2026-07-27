"""Shared experiment helpers for A2-K."""

from __future__ import annotations

import csv
import math
import statistics
import time
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

import torch

from cs336_systems.a2k.runtime import (
    is_oom,
    nvidia_smi_metadata,
    peak_memory_mib,
    quantiles,
    reset_peak_memory,
    seed_all,
    set_allocator_limit,
    synchronize,
    write_json,
)


def parse_dtype(name: str) -> torch.dtype:
    values = {"fp32": torch.float32, "float32": torch.float32, "bf16": torch.bfloat16}
    try:
        return values[name.lower()]
    except KeyError as exc:
        raise ValueError(f"unsupported dtype: {name}") from exc


def make_metadata(seed: int, command: str) -> dict[str, Any]:
    allocator = set_allocator_limit()
    seed_all(seed)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
    return {
        "seed": seed,
        "command": command,
        "allocator": allocator,
        "hardware": nvidia_smi_metadata(),
        "measurement": {
            "device": "cuda:0" if torch.cuda.is_available() else "cpu",
            "cuda_synchronize": True,
            "single_process": True,
        },
    }


def _event_time_ms(fn: Callable[[], Any]) -> float:
    if torch.cuda.is_available():
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        return float(start.elapsed_time(end))
    start = time.perf_counter()
    fn()
    return (time.perf_counter() - start) * 1000.0


def measure(
    fn: Callable[[], Any],
    *,
    warmup_ms: int = 100,
    rep_ms: int = 300,
    reset_grads: Iterable[torch.Tensor] | None = None,
) -> dict[str, Any]:
    """CUDA-event benchmark equivalent to ``triton.testing.do_bench``.

    The warmup and repetition arguments are milliseconds, matching Triton's
    public ``do_bench`` contract.  Quantiles are computed over the individual
    event measurements.
    """

    def clear_grads() -> None:
        if reset_grads is not None:
            for tensor in reset_grads:
                tensor.grad = None

    clear_grads()
    fn()
    synchronize()
    clear_grads()
    estimate = max(_event_time_ms(fn), 1e-3)
    warmups = max(1, math.ceil(warmup_ms / estimate))
    repeats = max(1, math.ceil(rep_ms / estimate))
    for _ in range(warmups):
        clear_grads()
        fn()
    synchronize()
    clear_grads()
    reset_peak_memory()
    values: list[float] = []
    for _ in range(repeats):
        clear_grads()
        values.append(_event_time_ms(fn))
    p20, p50, p80 = quantiles(values)
    return {
        "warmup_ms": warmup_ms,
        "rep_ms": rep_ms,
        "samples": len(values),
        "latency_ms_samples": [round(value, 6) for value in values],
        "p20_ms": p20,
        "p50_ms": p50,
        "p80_ms": p80,
        "mean_ms": statistics.fmean(values),
        "peak_allocated_mib": peak_memory_mib()[0],
        "peak_reserved_mib": peak_memory_mib()[1],
    }


def run_safe(
    output: Path,
    metadata: dict[str, Any],
    body: Callable[[], dict[str, Any]],
) -> int:
    """Run one isolated experiment and always emit a structured result."""

    payload: dict[str, Any] = dict(metadata)
    reset_peak_memory()
    try:
        payload.update(body())
        payload["status"] = "complete"
    except BaseException as exc:
        payload["status"] = "oom" if is_oom(exc) else "failed"
        payload["error_type"] = type(exc).__name__
        payload["error"] = (
            "OutOfMemoryError"
            if payload["status"] == "oom"
            else f"{type(exc).__name__}: {str(exc)[:240]}"
        )
        payload["peak_allocated_mib"], payload["peak_reserved_mib"] = peak_memory_mib()
    write_json(output, payload)
    return 0 if payload["status"] == "complete" else 1


def append_csv(path: Path, row: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    fields = list(row)
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        if not exists:
            writer.writeheader()
        writer.writerow(row)
