"""Equivalent measurement boundaries shared by A2-K attention benchmarks."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

import torch
import triton

from student_scripts.a2k.common import peak_memory

AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]


def _clear_gradients(*tensors: torch.Tensor) -> None:
    for tensor in tensors:
        tensor.grad = None


def measure_attention_phase(
    function: AttentionFunction,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    phase: str,
    *,
    warmup_ms: int = 100,
    rep_ms: int = 300,
) -> dict[str, Any]:
    """Measure an attention phase after one untimed compile/initialization call."""
    grad_output = torch.randn_like(q)
    retained_output: torch.Tensor | None = None
    cold_start_ms = 0.0

    if phase == "forward":

        def operation() -> None:
            with torch.no_grad():
                function(q, k, v)

    elif phase == "backward":
        torch.cuda.synchronize()
        cold_forward_start = time.perf_counter()
        retained_output = function(q, k, v)
        torch.cuda.synchronize()
        cold_start_ms += (time.perf_counter() - cold_forward_start) * 1000

        def operation() -> None:
            assert retained_output is not None
            torch.autograd.grad(
                retained_output,
                (q, k, v),
                grad_output,
                retain_graph=True,
            )

    elif phase == "forward_backward":

        def operation() -> None:
            _clear_gradients(q, k, v)
            output = function(q, k, v)
            output.backward(grad_output)

    else:
        raise ValueError(f"unknown phase: {phase}")

    torch.cuda.synchronize()
    cold_operation_start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    cold_start_ms += (time.perf_counter() - cold_operation_start) * 1000
    _clear_gradients(q, k, v)
    torch.cuda.reset_peak_memory_stats()
    p20, p50, p80 = triton.testing.do_bench(
        operation,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    allocated, reserved = peak_memory()
    return {
        "cold_start_ms": cold_start_ms,
        "latency_p20_ms": float(p20),
        "latency_p50_ms": float(p50),
        "latency_p80_ms": float(p80),
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
    }
