from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def checkpoint_plot(results: Path, assets: Path) -> None:
    rows = [row for row in read_csv(results / "checkpointing.csv") if row["context_length"] == "1024"]
    labels = [str(row["checkpoint_block_size"]) for row in rows]
    memory = [float(row["peak_allocated_mib"]) / 1024 for row in rows]
    latency = [float(row["step_time_ms_p50"]) for row in rows]
    fig, left = plt.subplots(figsize=(7.5, 4.5))
    right = left.twinx()
    left.bar(labels, memory, color="#4472c4", alpha=0.8, label="Peak allocated")
    right.plot(labels, latency, color="#c00000", marker="o", linewidth=2, label="Step p50")
    left.set_xlabel("Checkpoint block size (none = no checkpoint)")
    left.set_ylabel("Peak allocated memory (GiB)")
    right.set_ylabel("Train-step p50 (ms)")
    left.set_title("Activation checkpointing trade-off, context 1024")
    left.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(assets / "checkpoint_tradeoff.png", dpi=170)
    plt.close(fig)


def flash_plot(results: Path, assets: Path) -> None:
    rows = [row for row in read_csv(results / "flash_benchmark.csv") if row["phase"] == "forward" and row["head_dimension"] == "128" and row["status"] == "ok"]
    implementations = ("eager", "compiled", "triton")
    fig, axis = plt.subplots(figsize=(8, 4.8))
    for implementation in implementations:
        selected = sorted(
            (row for row in rows if row["implementation"] == implementation),
            key=lambda row: int(row["sequence_length"]),
        )
        if not selected:
            continue
        axis.plot(
            [int(row["sequence_length"]) for row in selected],
            [float(row["latency_ms_p50"]) for row in selected],
            marker="o",
            linewidth=2,
            label=implementation,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log", base=2)
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Forward p50 latency (ms)")
    axis.set_title("Causal BF16 attention, head dimension 128")
    axis.grid(alpha=0.25, which="both")
    axis.legend()
    fig.tight_layout()
    fig.savefig(assets / "flash_forward_latency.png", dpi=170)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render A2-K public figures from CSV results.")
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    checkpoint_plot(args.results, args.assets)
    flash_plot(args.results, args.assets)


if __name__ == "__main__":
    main()
