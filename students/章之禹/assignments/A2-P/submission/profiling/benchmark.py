"""End-to-end CUDA benchmark for the CS336 A2-P profiling tasks."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import torch

from profiling.common import (
    build_model,
    build_optimizer,
    finite_stats,
    hardware_metadata,
    make_batch,
    make_targets,
    patch_attention_annotations,
    peak_memory_mib,
    reset_peak_memory,
    resolve_device,
    run_step,
    sanitized_command,
    synchronize,
    dtype_from_name,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "medium", "large", "xl", "10b"), default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=("forward", "forward_backward", "train_step"), default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--annotate-attention", action="store_true")
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    if args.warmup < 0 or args.steps <= 0:
        raise ValueError("warmup must be non-negative and steps must be positive")
    device = resolve_device(args.device)
    amp_dtype = dtype_from_name(args.dtype)
    # Autocast changes selected operations, not the stored model parameters.
    # Keep parameters in FP32 for both sides of the mixed-precision comparison.
    parameter_dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    if args.annotate_attention:
        patch_attention_annotations()

    config = {
        "model_size": args.model_size,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "dtype": args.dtype,
        "autocast": bool(args.autocast),
        "parameter_dtype": "torch.float32",
        "compute_dtype": str(amp_dtype if args.autocast else torch.float32),
        "seed": args.seed,
    }
    try:
        model = build_model(
            args.model_size,
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            device=device,
            parameter_dtype=parameter_dtype,
        )
        model.train(args.mode != "forward")
        optimizer = None if args.mode == "forward" else build_optimizer(model)
        inputs = make_batch(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device,
            seed=args.seed + 1,
        )
        targets = (
            None
            if args.mode == "forward"
            else make_targets(inputs, vocab_size=args.vocab_size, seed=args.seed + 2)
        )
    except (RuntimeError, MemoryError) as error:
        return {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": sanitized_command("python profiling/benchmark.py", sys.argv[1:]),
            "peak_allocated_mib": peak_memory_mib(device)[0],
            "peak_reserved_mib": peak_memory_mib(device)[1],
        }

    def one_step() -> torch.Tensor | None:
        return run_step(
            model,
            optimizer,
            inputs,
            targets,
            mode=args.mode,
            amp_dtype=amp_dtype if args.autocast else torch.float32,
            device=device,
        )

    try:
        for _ in range(args.warmup):
            one_step()
            synchronize(device)
        synchronize(device)
        reset_peak_memory(device)
        timings: list[float] = []
        losses: list[float] = []
        for _ in range(args.steps):
            synchronize(device)
            started = time.perf_counter()
            loss = one_step()
            synchronize(device)
            timings.append((time.perf_counter() - started) * 1000.0)
            if loss is not None:
                losses.append(float(loss))
        peak_allocated, peak_reserved = peak_memory_mib(device)
        result: dict[str, Any] = {
            "schema_version": 1,
            "status": "complete",
            "config": config,
            "hardware": hardware_metadata(device),
            "command": sanitized_command("python profiling/benchmark.py", sys.argv[1:]),
            "timings_ms": timings,
            "stats": finite_stats(timings),
            "peak_allocated_mib": peak_allocated,
            "peak_reserved_mib": peak_reserved,
            "losses": losses,
        }
    except (RuntimeError, MemoryError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": sanitized_command("python profiling/benchmark.py", sys.argv[1:]),
            "peak_allocated_mib": peak_memory_mib(device)[0],
            "peak_reserved_mib": peak_memory_mib(device)[1],
        }
    return result


def main() -> int:
    args = parse_args()
    result = run_benchmark(args)
    rendered = json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False)
    print(rendered)
    if args.output:
        write_json(args.output, result)
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
