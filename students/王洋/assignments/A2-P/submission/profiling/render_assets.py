from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def benchmark_plot(results: Path, assets: Path) -> None:
    rows = [row for row in read_csv(results / "benchmark.csv") if row["status"] == "ok"]
    labels = [f"{row['mode']}\nw{row['warmup_steps']}" for row in rows]
    means = [float(row["mean_ms"]) for row in rows]
    errors = [float(row["sample_std_ms"]) for row in rows]
    fig, axis = plt.subplots(figsize=(8, 4.5))
    axis.bar(labels, means, yerr=errors, capsize=4, color="#4472c4")
    axis.set_ylabel("Latency per step (ms)")
    axis.set_title("Small model, batch 4, context 512, FP32")
    axis.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(assets / "benchmark_latency.png", dpi=170)
    plt.close(fig)


def profile_plot(results: Path, assets: Path) -> None:
    rows = read_csv(results / "profile" / "stage_summary.csv")
    wanted = {"forward", "backward", "optimizer", "attention/scores", "attention/softmax", "attention/value"}
    grouped: dict[str, dict[str, float]] = defaultdict(dict)
    for row in rows:
        if row["stage"] in wanted:
            grouped[row["run"]][row["stage"]] = float(row["cuda_device_time_sum_ms"])
    runs = sorted(grouped)
    stages = sorted(wanted)
    bottom = [0.0] * len(runs)
    fig, axis = plt.subplots(figsize=(10, 5))
    for stage in stages:
        values = [grouped[run].get(stage, 0.0) for run in runs]
        axis.bar(runs, values, bottom=bottom, label=stage)
        bottom = [left + value for left, value in zip(bottom, values, strict=True)]
    axis.set_ylabel("Summed CUDA device-event time (ms)")
    axis.set_title("Profiled train-step ranges (overlapping child ranges shown separately)")
    axis.tick_params(axis="x", rotation=35)
    axis.legend(fontsize=8, ncol=2)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets / "profile_stage_summary.png", dpi=170)
    plt.close(fig)


def memory_plot(results: Path, assets: Path) -> None:
    rows = read_csv(results / "memory" / "peaks.csv")
    labels = [f"{row['model_size']} c{row['context_length']}\n{row['mode']} {row['dtype']}" for row in rows]
    values = [float(row.get("peak_allocated_mib") or 0) for row in rows]
    colors = ["#70ad47" if row["status"] == "ok" else "#c00000" for row in rows]
    fig, axis = plt.subplots(figsize=(11, 5))
    axis.bar(labels, values, color=colors)
    axis.set_ylabel("Peak allocated memory (MiB)")
    axis.set_title("Memory profiling matrix (red indicates OOM)")
    axis.tick_params(axis="x", rotation=30)
    axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets / "memory_peak_summary.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render A2-P public figures from lightweight results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    benchmark_plot(args.results, args.assets)
    profile_plot(args.results, args.assets)
    memory_plot(args.results, args.assets)


if __name__ == "__main__":
    main()
