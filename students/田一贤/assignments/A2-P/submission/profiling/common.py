from __future__ import annotations

import contextlib
import statistics
import time

import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM

from .config import get_model_spec


def build_model(model_size: str, context_length: int, device: torch.device):
    spec = get_model_spec(model_size)
    return BasicsTransformerLM(
        vocab_size=spec.vocab_size,
        context_length=context_length,
        d_model=spec.d_model,
        num_layers=spec.num_layers,
        num_heads=spec.num_heads,
        d_ff=spec.d_ff,
    ).to(device)


def autocast_context(device: torch.device, dtype_name: str):
    if dtype_name == "bf16":
        return torch.autocast(device_type=device.type, dtype=torch.bfloat16)
    return contextlib.nullcontext()


def synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def loss_from_logits(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.flatten(0, -2), targets.flatten())


def summarize(samples_ms: list[float]) -> dict[str, float | list[float]]:
    mean = statistics.fmean(samples_ms)
    stdev = statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0
    return {
        "samples_ms": samples_ms,
        "mean_ms": mean,
        "stdev_ms": stdev,
        "cv": stdev / mean if mean else 0.0,
    }


def timed_step(fn, device: torch.device) -> float:
    synchronize(device)
    start = time.perf_counter_ns()
    fn()
    synchronize(device)
    return (time.perf_counter_ns() - start) / 1e6
