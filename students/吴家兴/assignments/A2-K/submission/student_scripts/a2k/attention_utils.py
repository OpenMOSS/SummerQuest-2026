"""Common input and phase timing helpers for attention experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Literal

import torch

from cs336_systems.a2k.runtime import benchmark_cuda


Phase = Literal["forward", "backward", "forward-backward"]


@dataclass
class AttentionInputs:
    q: torch.Tensor
    k: torch.Tensor
    v: torch.Tensor
    grad_output: torch.Tensor

    def clear_gradients(self) -> None:
        self.q.grad = None
        self.k.grad = None
        self.v.grad = None


def make_attention_inputs(
    *,
    sequence_length: int,
    head_dim: int,
    phase: Phase,
    seed: int,
    dtype: torch.dtype = torch.bfloat16,
) -> AttentionInputs:
    torch.manual_seed(seed)
    requires_grad = phase != "forward"
    shape = (1, sequence_length, head_dim)
    q = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    k = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    v = torch.randn(
        shape,
        device="cuda",
        dtype=dtype,
        requires_grad=requires_grad,
    )
    grad_output = torch.randn(shape, device="cuda", dtype=dtype)
    return AttentionInputs(q=q, k=k, v=v, grad_output=grad_output)


def run_phase_once(
    forward: Callable[[], torch.Tensor],
    inputs: AttentionInputs,
    phase: Phase,
) -> None:
    """Run one complete phase, used for cold-start measurement."""

    if phase == "forward":
        forward()
        return
    inputs.clear_gradients()
    output = forward()
    output.backward(inputs.grad_output)


def benchmark_attention_phase(
    forward: Callable[[], torch.Tensor],
    inputs: AttentionInputs,
    phase: Phase,
    *,
    warmup_ms: float,
    rep_ms: float,
) -> dict[str, Any]:
    """Measure forward, backward-only, or combined forward-backward."""

    if phase == "forward":
        return benchmark_cuda(
            forward,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )
    if phase == "forward-backward":

        def measured() -> None:
            inputs.clear_gradients()
            forward().backward(inputs.grad_output)

        return benchmark_cuda(
            measured,
            warmup_ms=warmup_ms,
            rep_ms=rep_ms,
        )

    state: dict[str, torch.Tensor] = {}

    def prepare() -> None:
        inputs.clear_gradients()
        state["output"] = forward()

    def measured_backward() -> None:
        state.pop("output").backward(inputs.grad_output)

    return benchmark_cuda(
        measured_backward,
        prepare=prepare,
        warmup_ms=warmup_ms,
        rep_ms=rep_ms,
    )
