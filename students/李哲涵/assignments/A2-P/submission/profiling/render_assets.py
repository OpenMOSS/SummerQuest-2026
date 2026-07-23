from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def render_benchmark(public_root: Path, assets: Path) -> None:
    rows = [
        row
        for row in _read_csv(public_root / "benchmark.csv")
        if row.get("status") == "success" and row.get("mean_ms")
    ]
    if not rows:
        return
    labels = [f"{row['mode']}\nwarmup {row['warmup']}" for row in rows]
    values = [float(row["mean_ms"]) for row in rows]
    errors = [float(row.get("sample_std_ms") or 0) for row in rows]
    figure, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(range(len(rows)), values, yerr=errors, color="#4472c4", capsize=4)
    axis.set_xticks(range(len(rows)), labels, rotation=15)
    axis.set_ylabel("time per step (ms)")
    axis.set_title("A2-P small model end-to-end benchmark")
    axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(assets / "benchmark_latency.png", dpi=150)
    plt.close(figure)


def _profile_events(trace_path: Path) -> list[dict[str, Any]]:
    try:
        return json.loads(trace_path.read_text()).get("traceEvents", [])
    except (OSError, json.JSONDecodeError):
        return []


def render_compute(raw_root: Path, assets: Path) -> None:
    traces = sorted((raw_root / "profile" / "traces").glob("*.json"))
    if not traces:
        return
    trace_path = traces[0]
    events = _profile_events(trace_path)
    selected = []
    names = (
        "forward",
        "backward",
        "optimizer",
        "attention/scores",
        "attention/softmax",
        "attention/value",
        "profile/measure",
    )
    for event in events:
        name = str(event.get("name", ""))
        if event.get("ph") == "X" and name in names and event.get("dur", 0) > 0:
            selected.append(event)
    if not selected:
        return
    start = min(float(event["ts"]) for event in selected)
    order = {name: index for index, name in enumerate(names)}
    colors = {
        "forward": "#4472c4",
        "backward": "#ed7d31",
        "optimizer": "#70ad47",
        "attention/scores": "#a5a5a5",
        "attention/softmax": "#ffc000",
        "attention/value": "#5b9bd5",
        "profile/measure": "#264478",
    }
    figure, axis = plt.subplots(figsize=(10, 5))
    for event in selected:
        name = str(event["name"])
        left = (float(event["ts"]) - start) / 1000
        width = float(event["dur"]) / 1000
        axis.barh(
            order[name],
            width,
            left=left,
            height=0.7,
            color=colors.get(name, "#999999"),
        )
    axis.set_yticks(range(len(names)), names)
    axis.set_xlabel("time from captured step (ms)")
    axis.set_title(f"torch.profiler phase timeline ({trace_path.stem})")
    axis.grid(axis="x", alpha=0.25)
    figure.tight_layout()
    figure.savefig(assets / "compute_timeline.png", dpi=150)
    plt.close(figure)


def _memory_points(snapshot_path: Path) -> tuple[list[float], list[float]]:
    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    traces = snapshot.get("device_traces", [])
    if not traces:
        return [], []
    current = 0
    points: list[tuple[float, float]] = []
    for event in traces[0]:
        action = event.get("action")
        size = int(event.get("size", 0))
        if action == "alloc":
            current += size
        elif action == "free_completed":
            current = max(0, current - size)
        elif action == "segment_free":
            current = max(0, current - size)
        points.append((float(event.get("time_us", 0)), current / (1024**2)))
    if not points:
        return [], []
    first = points[0][0]
    return [((time_us - first) / 1000) for time_us, _ in points], [value for _, value in points]


def render_memory(raw_root: Path, assets: Path) -> None:
    runs = []
    for path in sorted((raw_root / "memory" / "runs").glob("*.json")):
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        config = data.get("config", {})
        if config.get("model_size") == "xl" and data.get("snapshot_path"):
            runs.append((path, data))

    chosen: list[tuple[Path, dict[str, Any]]] = []
    for wanted_context, wanted_mode in ((128, "forward"), (2048, "train_step")):
        match = next(
            (
                item
                for item in runs
                if item[1].get("config", {}).get("context_length") == wanted_context
                and item[1].get("config", {}).get("mode") == wanted_mode
            ),
            None,
        )
        if match is not None:
            chosen.append(match)

    for run_path, data in chosen:
        snapshot_value = data["snapshot_path"]
        snapshot_path = raw_root / "memory" / "snapshots" / Path(snapshot_value).name
        if not snapshot_path.is_file():
            continue
        times, values = _memory_points(snapshot_path)
        if not times:
            continue
        config = data["config"]
        context = config["context_length"]
        mode = config["mode"]
        dtype = config["dtype"]
        figure, axis = plt.subplots(figsize=(9, 4.5))
        axis.plot(times, values, color="#4472c4", linewidth=1.1)
        axis.fill_between(times, values, color="#9dc3e6", alpha=0.5)
        axis.set_xlabel("time from snapshot start (ms)")
        axis.set_ylabel("active allocated memory (MiB)")
        axis.set_title(f"Active Memory Timeline — XL / context {context} / {mode} / {dtype}")
        axis.grid(alpha=0.25)
        figure.tight_layout()
        figure.savefig(assets / f"memory_timeline_s{context}_{mode}.png", dpi=150)
        plt.close(figure)


def render(raw_root: Path, public_root: Path, assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    render_benchmark(public_root, assets)
    render_compute(raw_root, assets)
    render_memory(raw_root, assets)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render lightweight A2-P assets")
    parser.add_argument("--raw-root", default="results/a2p")
    parser.add_argument("--public-root", default="results/a2p_public")
    parser.add_argument("--assets", default="results/a2p_assets")
    args = parser.parse_args()
    render(Path(args.raw_root), Path(args.public_root), Path(args.assets))
    print(f"Wrote assets to {args.assets}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
