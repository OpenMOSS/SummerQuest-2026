from __future__ import annotations

import math
import time

import torch


def dense_attention(q, k, v, causal=False):
    scores = q.float() @ k.float().transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        kj = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(qi[None] < kj[None], -1.0e9)
    probs = torch.softmax(scores, dim=-1)
    return (probs @ v.float()).to(q.dtype), torch.logsumexp(scores, dim=-1)


def native_attention(q, k, v, causal=False):
    """Explicit eager baseline that preserves the input compute dtype."""
    scores = q @ k.transpose(-1, -2) / math.sqrt(q.shape[-1])
    if causal:
        qi = torch.arange(q.shape[-2], device=q.device)[:, None]
        kj = torch.arange(k.shape[-2], device=q.device)[None, :]
        scores = scores.masked_fill(qi[None] < kj[None], -torch.inf)
    return torch.softmax(scores, dim=-1) @ v


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _percentile(samples: list[float], fraction: float) -> float:
    values = sorted(samples)
    index = (len(values) - 1) * fraction
    lo = int(index)
    hi = min(lo + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (index - lo)


def measure_attention_phase(
    implementation,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phase: str,
    *,
    warmup: int,
    steps: int,
) -> dict[str, object]:
    """Measure forward, backward-only, or the combined path."""
    device = q.device
    upstream = torch.randn_like(q)
    if phase == "forward":

        def one_step():
            with torch.no_grad():
                implementation(q, k, v)

    elif phase == "backward":
        prepared = implementation(q, k, v)

        def one_step():
            prepared.backward(upstream, retain_graph=True)
            q.grad = k.grad = v.grad = None

    elif phase == "forward_backward":

        def one_step():
            output = implementation(q, k, v)
            output.backward(upstream)
            q.grad = k.grad = v.grad = None

    else:
        raise ValueError(f"unknown phase: {phase}")

    for _ in range(warmup):
        one_step()
    _sync(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    samples: list[float] = []
    for _ in range(steps):
        _sync(device)
        started = time.perf_counter_ns()
        one_step()
        _sync(device)
        samples.append((time.perf_counter_ns() - started) / 1e6)
    allocated, reserved = peak_memory_mib(device)
    return {
        "samples_ms": samples,
        "p20_ms": _percentile(samples, 0.2),
        "p50_ms": _percentile(samples, 0.5),
        "p80_ms": _percentile(samples, 0.8),
        "warmup_steps": warmup,
        "measurement_steps": steps,
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
    }


def peak_memory_mib(device: torch.device) -> tuple[float, float]:
    if device.type != "cuda":
        return 0.0, 0.0
    return torch.cuda.max_memory_allocated(
        device
    ) / 2**20, torch.cuda.max_memory_reserved(device) / 2**20
