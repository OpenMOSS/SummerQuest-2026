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

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from runtime import MINIMUM_FREE_MIB, configure_single_gpu_allocator, peak_memory, synchronize


Implementation = Literal["eager", "compiled"]
Phase = Literal["forward", "forward_backward", "train_step"]
QUANTILES = (0.2, 0.5, 0.8)
VOCAB_SIZE = 10_000
MODEL_CONFIG = {
    "d_model": 768,
    "d_ff": 3072,
    "num_layers": 12,
    "num_heads": 12,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark one eager/compiled Stanford-small Transformer configuration.")
    parser.add_argument("--implementation", choices=("eager", "compiled"), required=True)
    parser.add_argument("--phase", choices=("forward", "forward_backward", "train_step"), required=True)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-free-mib", type=float, default=MINIMUM_FREE_MIB)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def build_model() -> BasicsTransformerLM:
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=512,
        **MODEL_CONFIG,
    ).cuda().train()


def autocast_context():
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def clear_parameter_gradients(parameters: tuple[torch.nn.Parameter, ...]) -> None:
    for parameter in parameters:
        parameter.grad = None


def make_phase_callable(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    phase: Phase,
) -> Callable[[], object]:
    if phase == "forward":
        def forward() -> torch.Tensor:
            with torch.no_grad(), autocast_context():
                return model(input_ids)

        return forward

    if phase == "forward_backward":
        def forward_backward() -> torch.Tensor:
            with autocast_context():
                loss = cross_entropy(model(input_ids), targets)
            loss.backward()
            return loss.detach()

        return forward_backward

    def train_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        with autocast_context():
            loss = cross_entropy(model(input_ids), targets)
        loss.backward()
        optimizer.step()
        return loss.detach()

    return train_step


def time_cuda_call(function: Callable[[], object]) -> tuple[object, float]:
    synchronize()
    start = perf_counter()
    output = function()
    synchronize()
    return output, (perf_counter() - start) * 1000


def linear_quantile(samples: list[float], quantile: float) -> float:
    ordered = sorted(samples)
    point = quantile * (len(ordered) - 1)
    lower = math.floor(point)
    upper = math.ceil(point)
    fraction = point - lower
    return (1 - fraction) * ordered[lower] + fraction * ordered[upper]


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
        raise ValueError("formal Transformer compile runs require warmup=100 ms and rep=300 ms")

    torch.backends.cuda.matmul.allow_tf32 = False
    torch.set_float32_matmul_precision("highest")
    allocator, environment = configure_single_gpu_allocator(args.minimum_free_mib)
    config = {
        "workload": "transformer_small",
        "implementation": args.implementation,
        "model_size": "small",
        **MODEL_CONFIG,
        "vocab_size": VOCAB_SIZE,
        "batch_size": 1,
        "context_length": 512,
        "dtype": "bf16",
        "parameter_dtype": "fp32",
        "phase": args.phase,
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "quantiles": list(QUANTILES),
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "compile_fullgraph": args.implementation == "compiled",
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

    failure_stage = "setup"
    try:
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)
        eager_model = build_model()
        optimizer = AdamW(eager_model.parameters(), lr=args.learning_rate)
        parameters = tuple(eager_model.parameters())
        input_ids = torch.randint(0, VOCAB_SIZE, (1, 512), device="cuda")
        targets = torch.randint(0, VOCAB_SIZE, (1, 512), device="cuda")
        model = torch.compile(eager_model, fullgraph=True) if args.implementation == "compiled" else eager_model
        phase_callable = make_phase_callable(
            model,
            optimizer,
            input_ids,
            targets,
            args.phase,
        )

        if args.implementation == "compiled":
            failure_stage = "cold_compile"
            clear_parameter_gradients(parameters)
            _, cold_start_ms = time_cuda_call(phase_callable)
            result["cold_start"] = {"total_ms": cold_start_ms}
            clear_parameter_gradients(parameters)

        failure_stage = "steady_state_benchmark"
        samples_ms = triton.testing.do_bench(
            phase_callable,
            warmup=args.warmup_ms,
            rep=args.rep_ms,
            grad_to_none=parameters if args.phase == "forward_backward" else None,
            return_mode="all",
        )
        p20_ms, p50_ms, p80_ms = (linear_quantile(samples_ms, quantile) for quantile in QUANTILES)

        failure_stage = "memory_measurement"
        clear_parameter_gradients(parameters)
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
