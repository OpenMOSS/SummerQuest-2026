from __future__ import annotations

import argparse
import statistics
import time
from pathlib import Path

import torch

from profiling.common import (
    RunConfig,
    append_jsonl,
    build_model,
    build_optimizer,
    cleanup_cuda,
    coefficient_of_variation,
    command_string,
    config_dict,
    make_batch,
    public_environment,
    require_cuda,
    run_model_step,
    set_seed,
    synchronize,
)


def benchmark(config: RunConfig, warmup: int, steps: int) -> dict:
    device = require_cuda()
    set_seed(config.seed)
    model = build_model(config, device)
    inputs, targets = make_batch(config, device)
    optimizer = build_optimizer(model, config)

    try:
        for _ in range(warmup):
            run_model_step(model, inputs, targets, config, optimizer)
            synchronize()

        raw_ms: list[float] = []
        losses: list[float] = []
        torch.cuda.reset_peak_memory_stats(device)
        for _ in range(steps):
            synchronize()
            start = time.perf_counter()
            loss = run_model_step(model, inputs, targets, config, optimizer)
            synchronize()
            raw_ms.append((time.perf_counter() - start) * 1_000)
            if loss is not None:
                losses.append(loss)

        return {
            "status": "ok",
            **config_dict(config),
            "warmup_steps": warmup,
            "measurement_steps": steps,
            "raw_ms": raw_ms,
            "mean_ms": statistics.fmean(raw_ms),
            "sample_std_ms": statistics.stdev(raw_ms) if len(raw_ms) > 1 else 0.0,
            "cv": coefficient_of_variation(raw_ms),
            "last_loss": losses[-1] if losses else None,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "command": command_string(),
            "environment": public_environment(),
        }
    finally:
        cleanup_cuda(model, inputs, targets, optimizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark a CS336 language-model step.")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        dtype=args.dtype,
        seed=args.seed,
    )
    try:
        result = benchmark(config, args.warmup, args.steps)
    except torch.cuda.OutOfMemoryError as exc:
        cleanup_cuda()
        result = {
            "status": "oom",
            **config_dict(config),
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "error_type": type(exc).__name__,
            "command": command_string(),
            "environment": public_environment(),
        }
    append_jsonl(args.output, result)
    print(result)


if __name__ == "__main__":
    main()
