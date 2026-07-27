"""Collect one stable train-step trace with torch.profiler."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile

from profiling.common import (
    build_model,
    build_optimizer,
    dtype_from_name,
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
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-size", choices=("small", "medium", "large", "xl", "10b"), default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp32", "bf16", "fp16"), default="fp32")
    parser.add_argument("--autocast", action="store_true")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--output", type=Path, required=True, help="summary JSON path")
    parser.add_argument("--trace-output", type=Path, required=True, help="Chrome trace path")
    parser.add_argument("--table-output", type=Path, required=True, help="operator CSV path")
    return parser.parse_args()


def run_profile(args: argparse.Namespace) -> dict[str, Any]:
    device = resolve_device(args.device)
    amp_dtype = dtype_from_name(args.dtype)
    # Keep stored parameters in FP32 for the autocast comparison.
    parameter_dtype = torch.float32
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    patch_attention_annotations()
    config = {
        "model_size": args.model_size,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": "train_step",
        "warmup_steps": args.warmup,
        "measurement_steps": 1,
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
        model.train()
        optimizer = build_optimizer(model)
        inputs = make_batch(
            batch_size=args.batch_size,
            context_length=args.context_length,
            vocab_size=args.vocab_size,
            device=device,
            seed=args.seed + 1,
        )
        targets = make_targets(inputs, vocab_size=args.vocab_size, seed=args.seed + 2)
    except (RuntimeError, MemoryError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "initialization",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": sanitized_command("python profiling/profile.py", sys.argv[1:]),
            "peak_allocated_mib": peak_memory_mib(device)[0],
            "peak_reserved_mib": peak_memory_mib(device)[1],
        }
        write_json(args.output, result)
        return result
    try:
        for _ in range(args.warmup):
            run_step(
                model,
                optimizer,
                inputs,
                targets,
                mode="train_step",
                amp_dtype=amp_dtype if args.autocast else torch.float32,
                device=device,
            )
            synchronize(device)
        synchronize(device)
        reset_peak_memory(device)
        activities = [ProfilerActivity.CPU]
        if device.type == "cuda":
            activities.append(ProfilerActivity.CUDA)
        with profile(
            activities=activities,
            record_shapes=True,
            profile_memory=True,
            with_stack=False,
        ) as profiler:
            # The actual warm-up is outside the profiler; this searchable
            # marker documents the boundary in the exported Chrome trace.
            with torch.profiler.record_function("profile/warmup"):
                pass
            with torch.profiler.record_function("profile/measure"):
                run_step(
                    model,
                    optimizer,
                    inputs,
                    targets,
                    mode="train_step",
                    amp_dtype=amp_dtype if args.autocast else torch.float32,
                    device=device,
                )
            synchronize(device)
            profiler.step()
    except (RuntimeError, MemoryError) as error:
        result = {
            "schema_version": 1,
            "status": "failed",
            "phase": "warmup_or_measure",
            "error_type": type(error).__name__,
            "error": str(error),
            "config": config,
            "hardware": hardware_metadata(device),
            "command": sanitized_command("python profiling/profile.py", sys.argv[1:]),
            "peak_allocated_mib": peak_memory_mib(device)[0],
            "peak_reserved_mib": peak_memory_mib(device)[1],
        }
        write_json(args.output, result)
        return result

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    profiler.export_chrome_trace(str(args.trace_output))
    stage_names = (
        "profile/warmup",
        "profile/measure",
        "zero_grad",
        "forward",
        "loss",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    )
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        # PyTorch 2.8 exposes device timings through ``device_time_total``;
        # older releases used the CUDA-specific alias.  Keep both paths so
        # the public summary has real GPU timings instead of silently writing
        # zeros when the profiler API changes.
        cuda_total = getattr(event, "device_time_total", None)
        if cuda_total is None:
            cuda_total = getattr(event, "cuda_time_total", 0.0)
        cuda_self = getattr(event, "self_device_time_total", None)
        if cuda_self is None:
            cuda_self = getattr(event, "self_cuda_time_total", 0.0)
        cpu_total = float(event.cpu_time_total)
        cuda_total = float(cuda_total)
        record_type = "operator"
        if event.key in stage_names:
            record_type = (
                "cpu_range"
                if cpu_total > 0.0
                else "gpu_annotation"
            )
        rows.append(
            {
                "name": event.key,
                "calls": int(event.count),
                "record_type": record_type,
                "cpu_total_us": cpu_total,
                "cpu_self_us": float(event.self_cpu_time_total),
                "cuda_total_us": cuda_total,
                "cuda_self_us": float(cuda_self),
                "cpu_memory_bytes": int(getattr(event, "cpu_memory_usage", 0)),
                "cuda_memory_bytes": int(getattr(event, "cuda_memory_usage", 0)),
            }
        )

    def stage_for(name: str) -> str:
        if name in stage_names:
            return name
        return "operator"

    ranked = sorted(
        rows,
        key=lambda row: max(float(row["cuda_total_us"]), float(row["cpu_total_us"])),
        reverse=True,
    )
    selected: list[dict[str, Any]] = []
    selected_names: set[str] = set()
    for row in rows:
        if row["name"] in stage_names:
            selected.append({**row, "stage": stage_for(row["name"])})
            selected_names.add(row["name"])
    for row in ranked:
        if row["name"] not in selected_names and len(selected) < 32:
            selected.append({**row, "stage": stage_for(row["name"])})
            selected_names.add(row["name"])
    args.table_output.parent.mkdir(parents=True, exist_ok=True)
    with args.table_output.open("w", newline="", encoding="utf-8") as handle:
        table_rows = selected
        fieldnames = list(table_rows[0]) if table_rows else ["name", "stage"]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(table_rows)
    peak_allocated, peak_reserved = peak_memory_mib(device)
    result = {
        "schema_version": 1,
        "status": "complete",
        "config": config,
        "hardware": hardware_metadata(device),
        "command": sanitized_command("python profiling/profile.py", sys.argv[1:]),
        "tool": "torch.profiler",
        "trace_output": str(args.trace_output),
        "table_output": str(args.table_output),
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "event_count": len(rows),
        "operator_summary": selected,
    }
    write_json(args.output, result)
    return result


def main() -> int:
    args = parse_args()
    result = run_profile(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["status"] == "complete" else 2


if __name__ == "__main__":
    raise SystemExit(main())
