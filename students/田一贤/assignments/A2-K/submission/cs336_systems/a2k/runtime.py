"""Small, dependency-light timing utilities for A2-K."""

from __future__ import annotations

import statistics
import time
from collections.abc import Callable

import torch


def synchronize_if_cuda(device: torch.device | str) -> None:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize()


def percentile(values: list[float], fraction: float) -> float:
    if not values:
        raise ValueError("cannot summarize an empty sample")
    ordered = sorted(values)
    index = (len(ordered) - 1) * fraction
    lo, hi = int(index), min(int(index) + 1, len(ordered) - 1)
    return ordered[lo] + (ordered[hi] - ordered[lo]) * (index - lo)


def time_callable(
    fn: Callable[[], object], device: torch.device | str, warmup: int, steps: int
) -> dict[str, object]:
    for _ in range(warmup):
        fn()
    synchronize_if_cuda(device)
    samples: list[float] = []
    for _ in range(steps):
        synchronize_if_cuda(device)
        start = time.perf_counter_ns()
        fn()
        synchronize_if_cuda(device)
        samples.append((time.perf_counter_ns() - start) / 1e6)
    return {
        "samples_ms": samples,
        "p20_ms": percentile(samples, 0.20),
        "p50_ms": percentile(samples, 0.50),
        "p80_ms": percentile(samples, 0.80),
        "mean_ms": statistics.fmean(samples),
        "stdev_ms": statistics.stdev(samples) if len(samples) > 1 else 0.0,
    }


__all__ = ["synchronize_if_cuda", "percentile", "time_callable"]
