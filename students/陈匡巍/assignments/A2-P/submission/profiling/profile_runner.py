"""Capture one stable train step with torch.profiler and stage annotations."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as functional
from torch.profiler import ProfilerActivity, profile, record_function

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MODEL_CONFIGS,
    configure_gpu,
    gpu_metadata,
    write_json,
)
from profiling.nvtx_ranges import install_attention_ranges


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("small", "medium"), required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


def stage_kernel_rows(trace_path: Path) -> list[dict[str, str | int | float]]:
    """Aggregate actual CUDA kernel events inside the three stage windows."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]

    def interval(name: str, category: str) -> tuple[float, float]:
        matches = [event for event in events if event.get("ph") == "X" and event.get("name") == name and event.get("cat") == category]
        if not matches:
            raise RuntimeError(f"missing {category} interval for {name}")
        event = max(matches, key=lambda item: float(item.get("dur", 0)))
        start = float(event["ts"])
        return start, start + float(event["dur"])

    stage_intervals = {
        "forward": interval("forward", "gpu_user_annotation"),
        "backward": interval("backward", "user_annotation"),
        "optimizer": interval(
            "Optimizer.step#AdamW.step",
            "gpu_user_annotation",
        ),
    }
    rows: list[dict[str, str | int | float]] = []
    for stage, (start, end) in stage_intervals.items():
        grouped: dict[str, dict[str, float]] = defaultdict(lambda: {"calls": 0, "cuda_total_us": 0.0})
        for event in events:
            if event.get("ph") != "X" or event.get("cat") != "kernel":
                continue
            timestamp = float(event.get("ts", -1))
            if start <= timestamp < end:
                values = grouped[str(event["name"])]
                values["calls"] += 1
                values["cuda_total_us"] += float(event.get("dur", 0))
        for name, values in sorted(
            grouped.items(),
            key=lambda item: item[1]["cuda_total_us"],
            reverse=True,
        )[:5]:
            rows.append(
                {
                    "event_type": f"{stage}_cuda_kernel",
                    "name": name,
                    "calls": int(values["calls"]),
                    "cpu_total_us": "",
                    "cpu_self_us": "",
                    "cuda_total_us": values["cuda_total_us"],
                    "cuda_self_us": values["cuda_total_us"],
                }
            )
    return rows


def main() -> int:
    args = parse_args()
    allocator = configure_gpu()
    gpu = gpu_metadata()
    install_attention_ranges()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    args.output_directory.mkdir(parents=True, exist_ok=True)

    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=args.context_length,
        **MODEL_CONFIGS[args.model_size],
    ).cuda()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, 10_000, (1, args.context_length), device="cuda")
    targets = torch.randint(0, 10_000, (1, args.context_length), device="cuda")

    def plain_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        logits = model(tokens)
        loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    for _ in range(3):
        plain_step()
        torch.cuda.synchronize()

    stage_events = {stage: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for stage in ("forward", "backward", "optimizer")}
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    ) as profiler:
        with record_function("profile/warmup"):
            torch.cuda.synchronize()
        with record_function("profile/measure"):
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward"):
                stage_events["forward"][0].record()
                logits = model(tokens)
                loss = functional.cross_entropy(
                    logits.reshape(-1, logits.shape[-1]),
                    targets.reshape(-1),
                )
                stage_events["forward"][1].record()
            with record_function("backward"):
                stage_events["backward"][0].record()
                loss.backward()
                stage_events["backward"][1].record()
            with record_function("optimizer"):
                stage_events["optimizer"][0].record()
                optimizer.step()
                stage_events["optimizer"][1].record()
        torch.cuda.synchronize()

    run_id = f"{args.model_size}_ctx{args.context_length}"
    trace_name = f"{run_id}_trace.json"
    trace_path = args.output_directory / trace_name
    profiler.export_chrome_trace(str(trace_path))

    required = {
        "profile/warmup",
        "profile/measure",
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    }
    aggregates: dict[tuple[str, str], dict[str, float]] = defaultdict(
        lambda: {
            "calls": 0,
            "cpu_total_us": 0.0,
            "cpu_self_us": 0.0,
            "device_total_us": 0.0,
            "device_self_us": 0.0,
        }
    )
    for event in profiler.key_averages():
        device_total = float(
            getattr(
                event,
                "device_time_total",
                getattr(event, "cuda_time_total", 0.0),
            )
        )
        device_self = float(
            getattr(
                event,
                "self_device_time_total",
                getattr(event, "self_cuda_time_total", 0.0),
            )
        )
        device_type = str(getattr(event, "device_type", "")).lower()
        if event.key in required:
            event_type = "range"
        elif "cuda" in device_type:
            event_type = "cuda_kernel"
        else:
            event_type = "torch_op"
        key = (event.key, event_type)
        aggregates[key] = {
            "calls": int(event.count),
            "cpu_total_us": float(event.cpu_time_total),
            "cpu_self_us": float(event.self_cpu_time_total),
            "device_total_us": device_total,
            "device_self_us": device_self,
        }

    ordered = sorted(
        aggregates.items(),
        key=lambda item: (
            item[0][0] not in required,
            -item[1]["device_total_us"],
            -item[1]["cpu_total_us"],
        ),
    )
    selected = [(key, values) for key, values in ordered if key[0] in required]
    selected.extend((key, values) for key, values in ordered if key[0] not in required)
    selected = selected[:40]

    summary_path = args.output_directory / f"{run_id}_summary.csv"
    with summary_path.open("w", encoding="utf-8", newline="") as handle:
        fields = [
            "model_size",
            "batch_size",
            "context_length",
            "dtype",
            "tool",
            "event_type",
            "name",
            "calls",
            "cpu_total_us",
            "cpu_self_us",
            "cuda_total_us",
            "cuda_self_us",
        ]
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for (name, event_type), values in selected:
            writer.writerow(
                {
                    "model_size": args.model_size,
                    "batch_size": 1,
                    "context_length": args.context_length,
                    "dtype": "fp32",
                    "tool": "torch.profiler",
                    "event_type": event_type,
                    "name": name,
                    "calls": values["calls"],
                    "cpu_total_us": values["cpu_total_us"],
                    "cpu_self_us": values["cpu_self_us"],
                    "cuda_total_us": values["device_total_us"],
                    "cuda_self_us": values["device_self_us"],
                }
            )
        for stage, (start, end) in stage_events.items():
            writer.writerow(
                {
                    "model_size": args.model_size,
                    "batch_size": 1,
                    "context_length": args.context_length,
                    "dtype": "fp32",
                    "tool": "cuda_event",
                    "event_type": "stage",
                    "name": stage,
                    "calls": 1,
                    "cpu_total_us": "",
                    "cpu_self_us": "",
                    "cuda_total_us": elapsed_ms(start, end) * 1000,
                    "cuda_self_us": "",
                }
            )
        for values in stage_kernel_rows(trace_path):
            writer.writerow(
                {
                    "model_size": args.model_size,
                    "batch_size": 1,
                    "context_length": args.context_length,
                    "dtype": "fp32",
                    "tool": "torch.profiler",
                    **values,
                }
            )

    metadata = {
        "run_id": run_id,
        "model_size": args.model_size,
        "batch_size": 1,
        "context_length": args.context_length,
        "mode": "train_step",
        "dtype": "fp32",
        "tool": "torch.profiler",
        "activities": ["CPU", "CUDA"],
        "unprofiled_warmup_steps": 3,
        "profiled_measurement_steps": 1,
        "trace_file": trace_name,
        "summary_file": summary_path.name,
        "command": (f"python -m profiling.profile_runner --model-size {args.model_size} --context-length {args.context_length}"),
        "allocator": allocator,
        "gpu": gpu,
    }
    write_json(args.output_directory / f"{run_id}_metadata.json", metadata)
    print(json.dumps(metadata, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
