from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt

from .common import read_json


STAGE_COLORS = {
    "forward": "#4C78A8",
    "backward": "#F58518",
    "optimizer": "#54A24B",
    "attention/scores": "#B279A2",
    "attention/softmax": "#E45756",
    "attention/value": "#72B7B2",
}


def profile_plot(summary_path: Path, output: Path) -> None:
    summary = read_json(summary_path)
    stage_cuda_ms = summary.get("stage_cuda_ms", {})
    preferred = ["forward", "backward", "optimizer"]
    by_name: dict[str, dict[str, Any]] = {}
    for row in summary["events"]:
        if row["name"] in preferred:
            existing = by_name.get(row["name"])
            if existing is None or row["cuda_total_us"] > existing["cuda_total_us"]:
                by_name[row["name"]] = row
    names = [name for name in preferred if name in stage_cuda_ms]
    values = [stage_cuda_ms[name] for name in names]
    calls = [1 for _ in names]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    bars = ax.barh(names[::-1], values[::-1], color=[STAGE_COLORS[name] for name in names[::-1]])
    ax.set_xlabel("Cumulative CUDA time (ms)")
    config = summary["config"]
    ax.set_title(f"torch.profiler stage summary: {config['model_size']}, context {config['context_length']}")
    for bar, value, count in zip(bars, values[::-1], calls[::-1], strict=True):
        ax.text(bar.get_width(), bar.get_y() + bar.get_height() / 2, f" {value:.2f} ms, {count} calls", va="center", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def trace_timeline(trace_path: Path, output: Path) -> None:
    trace = json.loads(trace_path.read_text())
    stages = ("forward", "backward", "optimizer")
    events = [
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "user_annotation" and event.get("ph") == "X" and event.get("name") in stages
    ]
    if not events:
        raise ValueError(f"No CPU stage ranges found in {trace_path}")
    measure = next(
        event
        for event in trace["traceEvents"]
        if event.get("cat") == "user_annotation" and event.get("name") == "profile/measure"
    )
    origin = measure["ts"]
    fig, ax = plt.subplots(figsize=(9, 3.6))
    for index, stage in enumerate(stages):
        stage_events = [event for event in events if event["name"] == stage]
        for event in stage_events:
            start_ms = (event["ts"] - origin) / 1000
            duration_ms = event["dur"] / 1000
            ax.broken_barh([(start_ms, duration_ms)], (index - 0.3, 0.6), facecolors=STAGE_COLORS[stage])
            ax.text(start_ms + duration_ms / 2, index, f"{duration_ms:.1f} ms", ha="center", va="center", color="white", fontsize=8)
    ax.set_yticks(range(len(stages)), stages)
    ax.set_xlabel("CPU wall time inside the profiled train step (ms)")
    ax.set_title("Representative train_step stage timeline")
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def memory_plot(summary_path: Path, output: Path) -> None:
    summary = read_json(summary_path)
    events = summary.get("transformer_blocks", {}).get("backward_memory_events", [])
    timeline = summary.get("active_memory_timeline", [])
    fig, ax = plt.subplots(figsize=(8, 4.2))
    if timeline:
        start = timeline[0]["time_us"]
        x = [(point["time_us"] - start) / 1000 for point in timeline]
        active = [point["active_bytes"] / 2**30 for point in timeline]
        ax.plot(x, active, color="#4C78A8", linewidth=1.25, label="active")
        ax.fill_between(x, active, alpha=0.18, color="#4C78A8")
        ax.set_xlabel("Time since first recorded allocator event (ms)")
        ax.legend()
    elif events:
        labels = [f"{event['block']}\n{event['event'].removeprefix('backward_')}" for event in events]
        allocated = [event["allocated_bytes"] / 2**30 for event in events]
        active = [event["active_bytes"] / 2**30 for event in events]
        x = range(len(events))
        ax.plot(x, allocated, marker="o", label="allocated")
        ax.plot(x, active, marker="s", label="active")
        ax.set_xticks(list(x), labels, rotation=45, ha="right", fontsize=7)
        ax.legend()
    else:
        memory = summary.get("memory", {})
        labels = ["peak active", "peak allocated", "peak reserved"]
        values = [
            (memory.get("peak_active_bytes") or 0) / 2**30,
            (memory.get("peak_allocated_bytes") or 0) / 2**30,
            (memory.get("peak_reserved_bytes") or 0) / 2**30,
        ]
        ax.bar(labels, values, color=["#4C78A8", "#F58518", "#54A24B"])
    config = summary["config"]
    ax.set_ylabel("GPU memory (GiB)")
    ax.set_title(f"Memory evidence: {config['model_size']}, context {config['context_length']}, {config['mode']}")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render compact, public A2-P evidence figures")
    subparsers = parser.add_subparsers(dest="kind", required=True)
    profile = subparsers.add_parser("profile")
    profile.add_argument("--summary", type=Path, required=True)
    profile.add_argument("--output", type=Path, required=True)
    trace = subparsers.add_parser("trace")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--output", type=Path, required=True)
    memory = subparsers.add_parser("memory")
    memory.add_argument("--summary", type=Path, required=True)
    memory.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.kind == "profile":
        profile_plot(args.summary, args.output)
    elif args.kind == "trace":
        trace_timeline(args.trace, args.output)
    else:
        memory_plot(args.summary, args.output)


if __name__ == "__main__":
    main()
