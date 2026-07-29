"""Render A2-P SVG assets from the lightweight submitted results."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib.pyplot as pyplot
from matplotlib.patches import Patch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--profile-csv", type=Path, required=True)
    parser.add_argument("--profile-trace", type=Path)
    parser.add_argument("--memory-metadata", type=Path, required=True)
    parser.add_argument("--memory-snapshot-128", type=Path)
    parser.add_argument("--memory-snapshot-2048", type=Path)
    parser.add_argument("--memory-snapshot-training", type=Path)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def benchmark_plot(rows: list[dict[str, str]], output: Path) -> None:
    successful = [row for row in rows if row["status"] == "ok" and row["warmup_steps"] == "5"]
    labels = [row["mode"].replace("_", "\n") for row in successful]
    means = [float(row["mean_ms"]) for row in successful]
    errors = [float(row["std_ms"]) for row in successful]
    figure, axis = pyplot.subplots(figsize=(6.8, 4.1))
    axis.bar(labels, means, yerr=errors, color="#2563eb", capsize=4)
    axis.set_ylabel("Latency (ms)")
    axis.set_title("Small model end-to-end latency (mean ± sample std)")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def profile_plot(rows: list[dict[str, str]], output: Path) -> None:
    stages = [row for row in rows if row["event_type"] == "stage" and row["model_size"] == "medium" and row["context_length"] == "1024"]
    labels = [row["name"] for row in stages]
    cuda_ms = [float(row["cuda_total_us"]) / 1000 for row in stages]
    figure, axis = pyplot.subplots(figsize=(6.8, 4.1))
    axis.bar(labels, cuda_ms, color=["#2563eb", "#dc2626", "#16a34a"])
    axis.set_ylabel("CUDA event time (ms)")
    axis.set_title("Representative train-step stage attribution")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def profile_timeline_plot(trace_path: Path, output: Path) -> None:
    """Render a cropped, public-safe view of an actual Chrome trace."""
    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = trace["traceEvents"]

    def longest(name: str, category: str) -> dict:
        matches = [event for event in events if event.get("ph") == "X" and event.get("name") == name and event.get("cat") == category]
        if not matches:
            raise RuntimeError(f"trace is missing {category}/{name}")
        return max(matches, key=lambda event: float(event.get("dur", 0)))

    measure = longest("profile/measure", "user_annotation")
    origin = float(measure["ts"])
    stage_events = {
        "forward": longest("forward", "user_annotation"),
        "backward": longest("backward", "user_annotation"),
        "optimizer": longest("Optimizer.step#AdamW.step", "gpu_user_annotation"),
    }
    end = max(float(event["ts"]) + float(event["dur"]) for event in stage_events.values())

    figure, axis = pyplot.subplots(figsize=(9.2, 4.5))
    stage_colors = {
        "forward": "#2563eb",
        "backward": "#dc2626",
        "optimizer": "#16a34a",
    }
    for stage, event in stage_events.items():
        start_ms = (float(event["ts"]) - origin) / 1000
        duration_ms = float(event["dur"]) / 1000
        axis.broken_barh(
            [(start_ms, duration_ms)],
            (42, 7),
            facecolors=stage_colors[stage],
        )
        axis.text(
            start_ms + duration_ms / 2,
            45.5,
            stage,
            color="white",
            fontsize=8,
            ha="center",
            va="center",
        )

    kernel_groups: dict[str, list[tuple[float, float]]] = {
        "GEMM": [],
        "softmax": [],
        "optimizer/elementwise": [],
        "other": [],
    }
    for event in events:
        if event.get("ph") != "X" or event.get("cat") != "kernel":
            continue
        timestamp = float(event.get("ts", -1))
        if not origin <= timestamp < end:
            continue
        name = str(event.get("name", "")).lower()
        if "cutlass" in name or "gemm" in name:
            group = "GEMM"
        elif "softmax" in name:
            group = "softmax"
        elif "multi_tensor" in name or "elementwise" in name:
            group = "optimizer/elementwise"
        else:
            group = "other"
        kernel_groups[group].append(
            (
                (timestamp - origin) / 1000,
                max(float(event.get("dur", 0)) / 1000, 0.002),
            )
        )
    kernel_colors = {
        "GEMM": "#7c3aed",
        "softmax": "#f59e0b",
        "optimizer/elementwise": "#0891b2",
        "other": "#94a3b8",
    }
    for group, spans in kernel_groups.items():
        if spans:
            axis.broken_barh(
                spans,
                (31, 7),
                facecolors=kernel_colors[group],
            )

    attention_lanes = {
        "attention/scores": (20, "#1d4ed8"),
        "attention/softmax": (10, "#d97706"),
        "attention/value": (0, "#0f766e"),
    }
    for name, (position, color) in attention_lanes.items():
        spans = [
            (
                (float(event["ts"]) - origin) / 1000,
                max(float(event["dur"]) / 1000, 0.002),
            )
            for event in events
            if event.get("ph") == "X" and event.get("cat") == "gpu_user_annotation" and event.get("name") == name and origin <= float(event["ts"]) < end
        ]
        if spans:
            axis.broken_barh(spans, (position, 6), facecolors=color)

    axis.set_xlim(0, (end - origin) / 1000)
    axis.set_ylim(-1, 51)
    axis.set_yticks(
        [45.5, 34.5, 23, 13, 3],
        [
            "stage range",
            "CUDA kernels",
            "attention/scores",
            "attention/softmax",
            "attention/value",
        ],
    )
    axis.set_xlabel("Time from measured train-step start (ms)")
    axis.set_title("Cropped torch.profiler timeline: medium, context 1024")
    axis.grid(axis="x", alpha=0.2)
    axis.legend(
        handles=[Patch(color=color, label=label) for label, color in kernel_colors.items()],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        ncols=4,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def memory_plot(run: dict, output: Path) -> None:
    samples = run["timeline_samples"]
    positions = list(range(len(samples)))
    allocated = [float(sample["allocated_mib"]) for sample in samples]
    reserved = [float(sample["reserved_mib"]) for sample in samples]
    figure, axis = pyplot.subplots(figsize=(7.6, 4.2))
    axis.plot(positions, allocated, label="active/allocated", color="#2563eb")
    axis.plot(positions, reserved, label="reserved", color="#dc2626")
    axis.set_xlabel("Transformer layer / stage order")
    axis.set_ylabel("MiB")
    axis.set_title(f"Active memory timeline: {run['row']['model_size']} ctx={run['row']['context_length']} {run['row']['mode']}")
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def snapshot_memory_plot(
    snapshot_path: Path,
    title: str,
    output: Path,
) -> None:
    with snapshot_path.open("rb") as handle:
        snapshot: dict[str, Any] = pickle.load(handle)
    events = snapshot["device_traces"][0]
    final_active = sum(int(block["size"]) for segment in snapshot["segments"] for block in segment["blocks"] if str(block["state"]).startswith("active"))
    final_reserved = sum(int(segment["total_size"]) for segment in snapshot["segments"])
    active_delta = sum(int(event.get("size", 0)) * (1 if event["action"] == "alloc" else -1) for event in events if event["action"] in {"alloc", "free_completed"})
    reserved_delta = sum(int(event.get("size", 0)) * (1 if event["action"] == "segment_alloc" else -1) for event in events if event["action"] in {"segment_alloc", "segment_free"})
    active = final_active - active_delta
    reserved = final_reserved - reserved_delta
    first_time = int(events[0]["time_us"])
    times_ms = [0.0]
    active_mib = [active / 1024**2]
    reserved_mib = [reserved / 1024**2]
    for event in events:
        action = event["action"]
        size = int(event.get("size", 0))
        if action == "alloc":
            active += size
        elif action == "free_completed":
            active -= size
        elif action == "segment_alloc":
            reserved += size
        elif action == "segment_free":
            reserved -= size
        else:
            continue
        times_ms.append((int(event["time_us"]) - first_time) / 1000)
        active_mib.append(active / 1024**2)
        reserved_mib.append(reserved / 1024**2)

    figure, axis = pyplot.subplots(figsize=(7.6, 4.2))
    axis.plot(times_ms, active_mib, label="active", color="#2563eb")
    axis.plot(times_ms, reserved_mib, label="reserved", color="#dc2626")
    axis.set_xlabel("Time since measured forward began (ms)")
    axis.set_ylabel("MiB")
    axis.set_title(title)
    axis.grid(alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    benchmark_plot(
        read_csv(args.benchmark_csv),
        args.output_directory / "benchmark_latency.svg",
    )
    profile_plot(
        read_csv(args.profile_csv),
        args.output_directory / "compute_profile_stages.svg",
    )
    if args.profile_trace is not None:
        profile_timeline_plot(
            args.profile_trace,
            args.output_directory / "compute_profile_timeline.svg",
        )
    metadata = json.loads(args.memory_metadata.read_text(encoding="utf-8"))
    forward_runs = [run for run in metadata["runs"] if run["row"]["model_size"] == "xl" and run["row"]["mode"] == "forward" and run["row"]["status"] == "ok"]
    for run in forward_runs:
        context = run["row"]["context_length"]
        snapshot = args.memory_snapshot_128 if context == 128 else args.memory_snapshot_2048
        output = args.output_directory / f"memory_timeline_xl_ctx{context}.svg"
        if snapshot is not None:
            snapshot_memory_plot(
                snapshot,
                f"Active memory timeline: XL ctx={context} forward",
                output,
            )
        else:
            memory_plot(run, output)
    if args.memory_snapshot_training is not None:
        snapshot_memory_plot(
            args.memory_snapshot_training,
            "Active memory timeline: Large ctx=128 train step (diagnostic)",
            args.output_directory / "memory_timeline_large_ctx128_train_step.svg",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
