from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from profiling.common import (
    RunConfig,
    build_model,
    build_optimizer,
    cleanup_cuda,
    command_string,
    config_dict,
    make_batch,
    public_environment,
    require_cuda,
    run_model_step,
    safe_write_json,
    set_seed,
    synchronize,
)


def _event_value(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def memory_profile(config: RunConfig, warmup: int, output_dir: Path) -> dict:
    device = require_cuda()
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{config.model_size}_ctx{config.context_length}_{config.mode}_{config.dtype}"
    model = build_model(config, device)
    inputs, targets = make_batch(config, device)
    optimizer = build_optimizer(model, config)
    try:
        for _ in range(warmup):
            run_model_step(model, inputs, targets, config, optimizer)
            synchronize()

        torch.cuda.reset_peak_memory_stats(device)
        with (
            torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                profile_memory=True,
                record_shapes=True,
                with_stack=True,
            ) as profiler,
            torch.profiler.record_function("memory/measure"),
        ):
            run_model_step(model, inputs, targets, config, optimizer)
            synchronize()

        timeline_path = output_dir / f"{run_name}.timeline.json"
        profiler.export_memory_timeline(str(timeline_path), device="cuda:0")

        stats = torch.cuda.memory_stats(device)
        result = {
            "status": "ok",
            "run": run_name,
            **config_dict(config),
            "warmup_steps": warmup,
            "measurement_steps": 1,
            "peak_active_mib": stats.get("active_bytes.all.peak", 0) / 2**20,
            "peak_allocated_mib": torch.cuda.max_memory_allocated(device) / 2**20,
            "peak_reserved_mib": torch.cuda.max_memory_reserved(device) / 2**20,
            "largest_allocation_mib": 0.0,
            "local_timeline_file": timeline_path.name,
            "command": command_string(),
            "environment": public_environment(),
        }
        memory_rows = []
        for event in profiler.key_averages():
            cuda_memory = _event_value(event, "self_device_memory_usage", "self_cuda_memory_usage")
            cpu_memory = _event_value(event, "self_cpu_memory_usage")
            memory_rows.append(
                {
                    "run": run_name,
                    "op_or_range": event.key,
                    "calls": int(event.count),
                    "self_cuda_memory_bytes": cuda_memory,
                    "self_cpu_memory_bytes": cpu_memory,
                }
            )
        memory_rows.sort(key=lambda row: abs(row["self_cuda_memory_bytes"]), reverse=True)
        positive_allocations = []
        timeline = profiler.mem_tl
        if timeline is not None:
            for event in timeline.timeline:
                if not isinstance(event, tuple) or len(event) < 4:
                    continue
                action = getattr(event[1], "name", str(event[1]))
                size = event[3]
                if action == "CREATE" and isinstance(size, int) and size > 0:
                    positive_allocations.append(size)
        if positive_allocations:
            result["largest_allocation_mib"] = max(positive_allocations) / 2**20
        with (output_dir / f"{run_name}.ops.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(memory_rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(memory_rows)
        return result
    finally:
        cleanup_cuda(model, inputs, targets, optimizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one A2-P memory profile.")
    parser.add_argument("--model-size", choices=["large", "xl"], default="xl")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--mode", choices=["forward", "train_step"], required=True)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    metadata_path = args.output_dir / f"{config.model_size}_ctx{config.context_length}_{config.mode}_{config.dtype}.metadata.json"
    try:
        result = memory_profile(config, args.warmup, args.output_dir)
    except torch.cuda.OutOfMemoryError as exc:
        allocated = torch.cuda.max_memory_allocated() / 2**20 if torch.cuda.is_available() else 0.0
        reserved = torch.cuda.max_memory_reserved() / 2**20 if torch.cuda.is_available() else 0.0
        cleanup_cuda()
        result = {
            "status": "oom",
            **config_dict(config),
            "warmup_steps": args.warmup,
            "error_type": type(exc).__name__,
            "peak_allocated_mib": allocated,
            "peak_reserved_mib": reserved,
            "command": command_string(),
            "environment": public_environment(),
        }
    safe_write_json(metadata_path, result)
    print(result)


if __name__ == "__main__":
    main()
