#!/usr/bin/env python3
"""Unified, synchronized end-to-end benchmark for A2-P."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path
from typing import Any

import torch

try:
    from .common import (
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        release_cuda,
        run_step,
        timing_statistics,
        write_json,
    )
    from .config import RunConfig, VALID_DTYPES, VALID_MODES
except ImportError:
    from common import (
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        release_cuda,
        run_step,
        timing_statistics,
        write_json,
    )
    from config import RunConfig, VALID_DTYPES, VALID_MODES


CSV_FIELDS = (
    "run_id",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "warmup_steps",
    "measurement_steps",
    "seed",
    "timings_ms",
    "mean_ms",
    "sample_std_ms",
    "cv",
    "min_ms",
    "max_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "first_loss",
    "last_loss",
)


def benchmark_once(config: RunConfig) -> dict[str, Any]:
    """Measure one configuration; initialization and data creation are excluded."""

    config.validate()
    model = build_model(config)
    inputs, targets = build_batch(config)
    optimizer = build_optimizer(model) if config.mode == "train_step" else None

    for _ in range(config.warmup_steps):
        run_step(
            model,
            inputs,
            targets,
            config.mode,
            config.dtype,
            optimizer,
        )
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    timings_ms: list[float] = []
    losses: list[float] = []
    for _ in range(config.measurement_steps):
        torch.cuda.synchronize()
        started_ns = time.perf_counter_ns()
        result = run_step(
            model,
            inputs,
            targets,
            config.mode,
            config.dtype,
            optimizer,
        )
        torch.cuda.synchronize()
        timings_ms.append((time.perf_counter_ns() - started_ns) / 1e6)
        if result.loss is not None:
            losses.append(result.loss)

    stats = timing_statistics(timings_ms)
    memory = memory_stats_mib()
    row: dict[str, Any] = {
        **config.as_dict(),
        "run_id": config.run_id,
        "timings_ms": json.dumps(
            [round(value, 6) for value in timings_ms],
            separators=(",", ":"),
        ),
        **{key: round(value, 6) for key, value in stats.items()},
        "peak_allocated_mib": round(memory["allocated_peak_mib"], 3),
        "peak_reserved_mib": round(memory["reserved_peak_mib"], 3),
        "first_loss": round(losses[0], 8) if losses else "",
        "last_loss": round(losses[-1], 8) if losses else "",
    }
    del model, inputs, targets, optimizer
    release_cuda()
    return row


def required_suite(seed: int, steps: int) -> list[RunConfig]:
    """The exact four rows required by the OpenMOSS A2-P benchmark section."""

    baseline = {
        "model_size": "small",
        "batch_size": 4,
        "context_length": 512,
        "dtype": "fp32",
        "measurement_steps": steps,
        "seed": seed,
    }
    return [
        RunConfig(**baseline, mode="train_step", warmup_steps=0),
        RunConfig(**baseline, mode="train_step", warmup_steps=5),
        RunConfig(**baseline, mode="forward", warmup_steps=5),
        RunConfig(**baseline, mode="forward_backward", warmup_steps=5),
    ]


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=VALID_MODES, default="train_step")
    parser.add_argument("--dtype", choices=VALID_DTYPES, default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--suite", choices=("single", "required"), default="single")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.suite == "required":
        configs = required_suite(args.seed, args.steps)
    else:
        configs = [
            RunConfig(
                model_size=args.model_size,
                batch_size=args.batch_size,
                context_length=args.context_length,
                mode=args.mode,
                dtype=args.dtype,
                warmup_steps=args.warmup,
                measurement_steps=args.steps,
                seed=args.seed,
            )
        ]

    rows = [benchmark_once(config) for config in configs]
    write_csv(args.output, rows)
    if args.metadata:
        write_json(
            args.metadata,
            {
                "schema_version": 1,
                "experiment": "end_to_end_benchmark",
                "timer": "time.perf_counter_ns",
                "synchronization": "before and after every measured CUDA step",
                "initialization_in_timed_region": False,
                "data_generation_in_timed_region": False,
                "environment": public_environment(),
                "runs": [config.as_dict() for config in configs],
                "output_file": args.output.name,
            },
        )
    for row in rows:
        print(
            f"{row['run_id']}: {row['mean_ms']:.3f} ms "
            f"(std={row['sample_std_ms']:.3f}, cv={row['cv']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
