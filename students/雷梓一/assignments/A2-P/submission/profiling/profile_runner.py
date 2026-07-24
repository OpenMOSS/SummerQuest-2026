from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.nn_utils import cross_entropy

from .benchmark import build_parser as benchmark_parser
from .benchmark import build_workload, execute_steps, make_step, synchronize
from .common import autocast_context, command_string, local_artifact_name, model_config_dict, range_context, software_metadata, utc_now, write_json
from .nvtx_ranges import annotated_attention


def build_parser() -> argparse.ArgumentParser:
    parent = benchmark_parser()
    parent.set_defaults(mode="train_step", warmup=5, steps=1)
    parser = argparse.ArgumentParser(description="Capture one stable A2-P train_step trace", parents=[parent], add_help=False)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--top-k", type=int, default=40)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.mode != "train_step":
        raise ValueError("The six required profiles must use mode=train_step")
    model, optimizer, tokens, targets, device = build_workload(args)
    if device.type != "cuda":
        raise RuntimeError("Compute profiling requires CUDA")
    warmup_step = make_step(args, model, optimizer, tokens, targets, device)
    stage_events = {stage: (torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)) for stage in ("forward", "backward", "optimizer")}

    def measured_step() -> float:
        optimizer.zero_grad(set_to_none=True)
        start, end = stage_events["forward"]
        start.record()
        with range_context("forward", device), autocast_context(device, args.dtype):
            logits = model(tokens)
            loss = cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
        end.record()

        start, end = stage_events["backward"]
        start.record()
        with range_context("backward", device):
            loss.backward()
        end.record()

        start, end = stage_events["optimizer"]
        start.record()
        with range_context("optimizer", device):
            optimizer.step()
        end.record()
        return float(loss.detach())
    args.trace_output.parent.mkdir(parents=True, exist_ok=True)

    with annotated_attention():
        execute_steps(warmup_step, args.warmup, device, "profile/warmup")
        activities = [torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA]
        with torch.profiler.profile(
            activities=activities,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as prof:
            # The actual warm-up has completed before collection. Keep a tiny
            # boundary marker in the exported trace without polluting the
            # single-step op/kernel aggregates with warm-up work.
            with range_context("profile/warmup", device):
                synchronize(device)
            execute_steps(measured_step, 1, device, "profile/measure")
            prof.step()
        synchronize(device)

    prof.export_chrome_trace(str(args.trace_output))
    stage_cuda_ms = {stage: events[0].elapsed_time(events[1]) for stage, events in stage_events.items()}
    rows = []
    for item in prof.key_averages():
        cuda_total = float(getattr(item, "device_time_total", 0.0))
        cuda_self = float(getattr(item, "self_device_time_total", 0.0))
        rows.append(
            {
                "name": item.key,
                "calls": int(item.count),
                "cpu_total_us": float(item.cpu_time_total),
                "cpu_self_us": float(item.self_cpu_time_total),
                "cuda_total_us": cuda_total,
                "cuda_self_us": cuda_self,
            }
        )
    rows.sort(key=lambda row: (row["cuda_total_us"], row["cpu_total_us"]), reverse=True)
    required_ranges = {
        "profile/warmup",
        "profile/measure",
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
    }
    selected = rows[: args.top_k]
    selected_names = {row["name"] for row in selected}
    selected.extend(row for row in rows if row["name"] in required_ranges - selected_names)
    # Very short NVTX/record_function boundary markers can be omitted from
    # key_averages on some profiler builds even though they are in the trace.
    if "profile/warmup" not in {row["name"] for row in selected}:
        selected.append(
            {
                "name": "profile/warmup",
                "calls": 1,
                "cpu_total_us": 0.0,
                "cpu_self_us": 0.0,
                "cuda_total_us": 0.0,
                "cuda_self_us": 0.0,
            }
        )
    summary = {
        "schema_version": 1,
        "status": "ok",
        "timestamp_utc": utc_now(),
        "tool": "torch.profiler",
        "command": command_string(),
        "trace_file": local_artifact_name(args.trace_output),
        "summary_file": local_artifact_name(args.summary_output),
        "config": {
            "model_size": args.model_size,
            **model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "mode": args.mode,
            "warmup": args.warmup,
            "profiled_steps": 1,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "software": software_metadata(device),
        "stage_cuda_ms": stage_cuda_ms,
        "events": selected,
    }
    write_json(args.summary_output, summary)
    print(f"wrote {args.trace_output} and {args.summary_output}")


if __name__ == "__main__":
    main()
