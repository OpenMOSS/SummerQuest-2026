from __future__ import annotations

import argparse
import json
import math
import shlex
import sys
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from time import perf_counter
from typing import Any, Literal

import torch
import triton.testing

from cs336_systems.a2k.attention import explicit_attention
from cs336_systems.a2k.flash_attention_triton import (
    KEY_TILE_SIZE,
    NUM_STAGES,
    NUM_WARPS,
    QUERY_TILE_SIZE,
    FlashAttentionTritonFunction,
)
from runtime import (
    MINIMUM_FREE_MIB,
    configure_single_gpu_allocator,
    peak_memory,
    synchronize,
)


AttentionFunction = Callable[[torch.Tensor, torch.Tensor, torch.Tensor, bool], torch.Tensor]
Implementation = Literal["eager", "compiled", "triton"]
Phase = Literal["forward", "backward", "forward_backward"]
QUANTILES = (0.2, 0.5, 0.8)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one attention configuration.")
    parser.add_argument("--sequence-length", type=int, required=True)
    parser.add_argument("--head-dim", type=int, required=True)
    parser.add_argument("--implementation", choices=("eager", "compiled", "triton"), required=True)
    parser.add_argument(
        "--phase",
        choices=("forward", "backward", "forward_backward"),
        required=True,
    )
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-mib", type=float, default=MINIMUM_FREE_MIB)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_attention(implementation: Implementation) -> AttentionFunction:
    if implementation == "eager":
        return explicit_attention
    if implementation == "compiled":
        return torch.compile(explicit_attention, fullgraph=True)
    return FlashAttentionTritonFunction.apply


def time_cuda_call(function: Callable[[], object]) -> tuple[object, float]:
    synchronize()
    start = perf_counter()
    output = function()
    synchronize()
    return output, (perf_counter() - start) * 1000


def clear_gradients(tensors: tuple[torch.Tensor, ...]) -> None:
    for tensor in tensors:
        tensor.grad = None


def linear_quantile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    point = quantile * (len(ordered) - 1)
    lower = math.floor(point)
    upper = math.ceil(point)
    fraction = point - lower
    return (1 - fraction) * ordered[lower] + fraction * ordered[upper]


def measure_cold_start(
    attention: AttentionFunction,
    phase: Phase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
) -> dict[str, float]:
    if phase == "forward":
        _, forward_ms = time_cuda_call(lambda: attention(q, k, v, True))
        return {"total_ms": forward_ms, "forward_ms": forward_ms}

    if phase == "backward":
        output, forward_ms = time_cuda_call(lambda: attention(q, k, v, True))
        _, backward_ms = time_cuda_call(lambda: output.backward(grad_output, retain_graph=True))
        clear_gradients((q, k, v))
        return {
            "total_ms": forward_ms + backward_ms,
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
        }

    def forward_backward() -> None:
        attention(q, k, v, True).backward(grad_output)

    _, total_ms = time_cuda_call(forward_backward)
    clear_gradients((q, k, v))
    return {"total_ms": total_ms}


def make_phase_callable(
    attention: AttentionFunction,
    phase: Phase,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    grad_output: torch.Tensor,
) -> Callable[[], object]:
    if phase == "forward":
        return lambda: attention(q, k, v, True)

    if phase == "backward":
        output = attention(q, k, v, True)
        return lambda: output.backward(grad_output, retain_graph=True)

    return lambda: attention(q, k, v, True).backward(grad_output)


def write_result(path: Path, result: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))


def main() -> None:
    args = parse_args()
    if args.warmup_ms != 100 or args.rep_ms != 300:
        raise ValueError("formal attention runs require warmup=100 ms and rep=300 ms")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    allocator, environment = configure_single_gpu_allocator(args.minimum_free_mib)

    config = {
        "implementation": args.implementation,
        "batch_size": 1,
        "sequence_length": args.sequence_length,
        "head_dim": args.head_dim,
        "dtype": "bf16",
        "is_causal": True,
        "phase": args.phase,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "quantiles": list(QUANTILES),
        "seed": args.seed,
        "triton_launch": (
            {
                "query_tile_size": QUERY_TILE_SIZE,
                "key_tile_size": KEY_TILE_SIZE,
                "num_warps": NUM_WARPS,
                "num_stages": NUM_STAGES,
            }
            if args.implementation == "triton"
            else None
        ),
    }
    result: dict[str, Any] = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": f"uv run python {shlex.join(sys.argv)}",
        "config": config,
        "environment": environment,
        "allocator": allocator,
        "status": "running",
        "failure_stage": None,
        "error_type": None,
    }

    failure_stage = "input_setup"
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        shape = (1, args.sequence_length, args.head_dim)
        q = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        k = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        v = torch.randn(shape, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad_output = torch.randn_like(q)
        attention = build_attention(args.implementation)

        if args.implementation in ("compiled", "triton"):
            failure_stage = "cold_compile"
            result["cold_start"] = measure_cold_start(
                attention,
                args.phase,
                q,
                k,
                v,
                grad_output,
            )

        failure_stage = "steady_state_setup"
        phase_callable = make_phase_callable(
            attention,
            args.phase,
            q,
            k,
            v,
            grad_output,
        )
        clear_gradients((q, k, v))

        failure_stage = "steady_state_benchmark"
        samples_ms = triton.testing.do_bench(
            phase_callable,
            warmup=args.warmup_ms,
            rep=args.rep_ms,
            grad_to_none=(q, k, v) if args.phase != "forward" else None,
            return_mode="all",
        )
        p20_ms, p50_ms, p80_ms = (linear_quantile(samples_ms, quantile) for quantile in QUANTILES)

        failure_stage = "memory_measurement"
        clear_gradients((q, k, v))
        synchronize()
        torch.cuda.reset_peak_memory_stats(0)
        measured_output = phase_callable()
        synchronize()

        result.update(
            {
                "status": "ok",
                "timing": {
                    "timer": "triton.testing.do_bench",
                    "warmup_ms": args.warmup_ms,
                    "rep_ms": args.rep_ms,
                    "quantiles": list(QUANTILES),
                    "measurement_count": len(samples_ms),
                    "p20_ms": p20_ms,
                    "p50_ms": p50_ms,
                    "p80_ms": p80_ms,
                },
                "memory": {
                    "peak_scope": "one steady-state phase invocation",
                    **peak_memory(),
                },
            }
        )
        del measured_output
    except torch.OutOfMemoryError:
        result.update(
            {
                "status": "oom",
                "failure_stage": failure_stage,
                "error_type": "OutOfMemoryError",
                "memory": {
                    "peak_scope": "process_until_failure",
                    **peak_memory(),
                },
            }
        )

    write_result(args.output, result)


if __name__ == "__main__":
    main()
