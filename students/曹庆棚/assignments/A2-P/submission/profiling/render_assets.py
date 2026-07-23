from __future__ import annotations

import argparse
import json
import pickle
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib.pyplot as plt


STAGE_COLORS = {
    "zero_grad": "#94a3b8",
    "forward": "#2563eb",
    "loss": "#7c3aed",
    "backward": "#dc2626",
    "optimizer": "#059669",
    "attention/scores": "#0ea5e9",
    "attention/softmax": "#f59e0b",
    "attention/value": "#14b8a6",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render lightweight, sanitized A2-P evidence figures")
    parser.add_argument("--profile-trace", type=Path, required=True)
    parser.add_argument("--forward-snapshot", type=Path, required=True)
    parser.add_argument("--train-snapshot", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def _save_figure(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, format="png", dpi=125, bbox_inches="tight", metadata={"Software": "matplotlib"})
    plt.close()


def render_profile(trace_path: Path, output: Path) -> None:
    payload = json.loads(trace_path.read_text(encoding="utf-8"))
    events = payload.get("traceEvents", [])
    measure = next(event for event in events if isinstance(event, dict) and event.get("ph") == "X" and event.get("name") == "profile/measure")
    start = float(measure["ts"])
    end = start + float(measure["dur"])
    duration_ms = (end - start) / 1000.0

    stage_rows: dict[str, list[tuple[float, float]]] = defaultdict(list)
    kernel_rows: list[tuple[float, float]] = []
    for event in events:
        if not isinstance(event, dict) or event.get("ph") != "X":
            continue
        event_start = float(event.get("ts", 0.0) or 0.0)
        event_end = event_start + float(event.get("dur", 0.0) or 0.0)
        if event_start < start or event_start >= end:
            continue
        name = str(event.get("name", ""))
        category = str(event.get("cat", "")).lower()
        clipped_end = min(event_end, end)
        row = ((event_start - start) / 1000.0, (clipped_end - event_start) / 1000.0)
        if name in STAGE_COLORS:
            stage_rows[name].append(row)
        elif "kernel" in category:
            kernel_rows.append(row)

    primary = ("zero_grad", "forward", "loss", "backward", "optimizer")
    attention = ("attention/scores", "attention/softmax", "attention/value")
    fig, axes = plt.subplots(3, 1, figsize=(12, 5.8), sharex=True, gridspec_kw={"height_ratios": [2, 2, 1]})

    for axis, names, title in ((axes[0], primary, "Training stages"), (axes[1], attention, "Attention subranges")):
        ticks = []
        labels = []
        for index, name in enumerate(names):
            y = len(names) - index
            for interval in stage_rows.get(name, []):
                axis.broken_barh([interval], (y - 0.35, 0.7), facecolors=STAGE_COLORS[name])
            ticks.append(y)
            labels.append(name)
        axis.set_yticks(ticks, labels)
        axis.set_title(title, loc="left", fontsize=10, fontweight="bold")
        axis.grid(axis="x", alpha=0.2)

    kernel_stride = max(len(kernel_rows) // 2000, 1)
    for interval in kernel_rows[::kernel_stride]:
        axes[2].broken_barh([interval], (0.2, 0.6), facecolors="#334155", alpha=0.65)
    axes[2].set_yticks([0.5], ["CUDA kernels"])
    axes[2].set_xlabel("Relative time within the measured train_step (ms)")
    axes[2].grid(axis="x", alpha=0.2)
    axes[2].set_xlim(0, duration_ms)
    fig.suptitle("Representative torch.profiler timeline: medium, context 512, batch 4, FP32", fontsize=12)
    fig.subplots_adjust(top=0.89, bottom=0.16, hspace=0.28)
    fig.text(
        0.5,
        0.025,
        "Chrome Trace rendered on a relative axis; absolute timestamps, process IDs, device IDs and paths are omitted.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    _save_figure(output)


def _current_active_bytes(snapshot: dict[str, Any]) -> int:
    total = 0
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                total += int(block.get("size", 0) or 0)
    return total


def _memory_series(snapshot: dict[str, Any]) -> tuple[list[int], list[float], list[float], float]:
    traces = snapshot.get("device_traces", [])
    events = traces[0] if traces else []
    deltas = []
    for event in events:
        action = event.get("action")
        size = int(event.get("size", 0) or 0)
        if action == "alloc":
            deltas.append(size)
        elif action == "free_completed":
            deltas.append(-size)
        else:
            deltas.append(0)

    ending = _current_active_bytes(snapshot)
    active = ending - sum(deltas)
    indices = [0]
    active_gib = [active / (1024**3)]
    for index, delta in enumerate(deltas, start=1):
        active += delta
        indices.append(index)
        active_gib.append(active / (1024**3))
    progress = [100.0 * index / max(len(deltas), 1) for index in indices]
    return indices, progress, active_gib, max(active_gib, default=0.0)


def render_memory(snapshot_path: Path, output: Path, title: str, subtitle: str) -> None:
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    _indices, progress, active_gib, peak = _memory_series(snapshot)

    fig, axis = plt.subplots(figsize=(11, 4.6))
    axis.fill_between(progress, active_gib, color="#2563eb", alpha=0.22, step="post")
    axis.plot(progress, active_gib, color="#1d4ed8", linewidth=1.15, drawstyle="steps-post")
    axis.axhline(peak, color="#dc2626", linestyle="--", linewidth=0.9, label=f"Trace peak: {peak:.2f} GiB")
    axis.set_xlim(0, 100)
    axis.set_ylim(bottom=0)
    axis.set_xlabel("Progress through recorded allocator events (%)")
    axis.set_ylabel("Active allocated memory (GiB)")
    fig.suptitle(title, x=0.08, y=0.97, ha="left", fontsize=12, fontweight="bold")
    fig.text(0.08, 0.92, subtitle, fontsize=9, color="#475569")
    axis.grid(alpha=0.2)
    axis.legend(loc="best", frameon=False)
    fig.text(
        0.5,
        0.025,
        "Derived from PyTorch memory history; allocation addresses, stack paths, process and device identifiers are omitted.",
        ha="center",
        fontsize=8,
        color="#475569",
    )
    fig.subplots_adjust(top=0.85, bottom=0.19, left=0.09, right=0.98)
    _save_figure(output)


def main() -> int:
    args = parse_args()
    render_profile(args.profile_trace, args.output_dir / "compute_profile_timeline.png")
    render_memory(
        args.forward_snapshot,
        args.output_dir / "memory_forward_timeline.png",
        "Active Memory Timeline: XL forward",
        "Context 2048, batch 4, FP32; warm-up completed before memory history was enabled.",
    )
    render_memory(
        args.train_snapshot,
        args.output_dir / "memory_train_timeline.png",
        "Active Memory Timeline: XL train_step",
        "Context 128, batch 4, FP32; includes forward, backward and AdamW update.",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
