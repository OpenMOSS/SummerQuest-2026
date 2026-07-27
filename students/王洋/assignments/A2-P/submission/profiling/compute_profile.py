from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch

from profiling.common import (
    RunConfig,
    autocast_context,
    build_model,
    build_optimizer,
    cleanup_cuda,
    command_string,
    config_dict,
    language_model_loss,
    make_batch,
    public_environment,
    require_cuda,
    run_model_step,
    safe_write_json,
    set_seed,
    synchronize,
)
from profiling.nvtx_ranges import instrument_attention


def _event_value(event, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def timed_stages(model, inputs, targets, config, optimizer, repeats: int = 5) -> dict[str, list[float]]:
    timings = {"forward": [], "backward": [], "optimizer": [], "train_step": []}
    for _ in range(repeats):
        optimizer.zero_grad(set_to_none=True)
        events = {name: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for name in timings}
        events["train_step"][0].record()
        events["forward"][0].record()
        with autocast_context(config.dtype):
            logits = model(inputs)
            loss = language_model_loss(logits, targets)
        events["forward"][1].record()
        events["backward"][0].record()
        loss.backward()
        events["backward"][1].record()
        events["optimizer"][0].record()
        optimizer.step()
        events["optimizer"][1].record()
        events["train_step"][1].record()
        synchronize()
        for name, (start, end) in events.items():
            timings[name].append(float(start.elapsed_time(end)))
    return timings


def profile_once(config: RunConfig, warmup: int, output_dir: Path) -> None:
    device = require_cuda()
    set_seed(config.seed)
    output_dir.mkdir(parents=True, exist_ok=True)
    run_name = f"{config.model_size}_ctx{config.context_length}_{config.dtype}"
    model = build_model(config, device)
    inputs, targets = make_batch(config, device)
    optimizer = build_optimizer(model, config)

    try:
        with instrument_attention():
            for _ in range(warmup):
                run_model_step(model, inputs, targets, config, optimizer)
                synchronize()

            stage_timings_ms = timed_stages(model, inputs, targets, config, optimizer)

            activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
            with torch.profiler.profile(
                activities=activities,
                record_shapes=True,
                with_stack=False,
                profile_memory=False,
            ) as profiler:
                with torch.profiler.record_function("profile/warmup"):
                    pass
                with torch.profiler.record_function("profile/measure"):
                    run_model_step(model, inputs, targets, config, optimizer)
                    synchronize()

        trace_path = output_dir / f"{run_name}.trace.json"
        profiler.export_chrome_trace(str(trace_path))

        rows = []
        for event in profiler.key_averages():
            rows.append(
                {
                    "run": run_name,
                    "model_size": config.model_size,
                    "context_length": config.context_length,
                    "dtype": config.dtype,
                    "op_or_range": event.key,
                    "calls": int(event.count),
                    "cpu_time_total_us": _event_value(event, "cpu_time_total"),
                    "cuda_time_total_us": _event_value(event, "device_time_total", "cuda_time_total"),
                    "self_cpu_time_total_us": _event_value(event, "self_cpu_time_total"),
                    "self_cuda_time_total_us": _event_value(event, "self_device_time_total", "self_cuda_time_total"),
                }
            )
        rows.sort(key=lambda row: row["cuda_time_total_us"], reverse=True)
        with (output_dir / f"{run_name}.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(rows)

        safe_write_json(
            output_dir / f"{run_name}.metadata.json",
            {
                "status": "ok",
                "run": run_name,
                **config_dict(config),
                "tool": "torch.profiler",
                "external_warmup_steps": warmup,
                "captured_warmup_steps": 0,
                "captured_measurement_steps": 1,
                "local_trace_file": trace_path.name,
                "stage_timings_ms": stage_timings_ms,
                "command": command_string(),
                "environment": public_environment(),
            },
        )
    finally:
        cleanup_cuda(model, inputs, targets, optimizer)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one A2-P train-step profile.")
    parser.add_argument("--model-size", choices=["small", "medium"], required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", choices=["fp32", "bf16"], default="bf16")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = RunConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode="train_step",
        dtype=args.dtype,
        seed=args.seed,
    )
    profile_once(config, args.warmup, args.output_dir)


if __name__ == "__main__":
    main()
