from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def read(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results/a2k_submission"))
    parser.add_argument("--assets", type=Path, default=Path("results/a2k_assets"))
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)

    checkpoint = [row for row in read(args.results / "checkpointing.csv") if row["context_length"] == "1024" and row["status"] == "success"]
    labels = [row["checkpoint_block_size"] for row in checkpoint]
    memory = [float(row["peak_allocated_mib"]) / 1024 for row in checkpoint]
    latency = [float(row["step_time_ms_p50"]) for row in checkpoint]
    figure, left = plt.subplots(figsize=(7, 4))
    right = left.twinx()
    left.bar(labels, memory, color="#4c78a8", alpha=0.8)
    right.plot(labels, latency, color="#e45756", marker="o")
    left.set(xlabel="Checkpoint block size", ylabel="Peak allocated (GiB)")
    right.set_ylabel("Step p50 (ms)")
    figure.tight_layout()
    figure.savefig(args.assets / "checkpoint_memory_latency.png", dpi=160)
    plt.close(figure)

    flash = [row for row in read(args.results / "flash_benchmark.csv") if row["phase"] == "forward" and row["head_dim"] == "128" and row["status"] == "success" and row["sequence_length"] != "16384"]
    implementations = ("eager", "compiled", "triton")
    sequences = sorted({int(row["sequence_length"]) for row in flash})
    figure, axis = plt.subplots(figsize=(7, 4))
    for implementation in implementations:
        values = {int(row["sequence_length"]): float(row["p50_ms"]) for row in flash if row["implementation"] == implementation}
        axis.plot(sequences, [values.get(sequence, float("nan")) for sequence in sequences], marker="o", label=implementation)
    axis.set(xlabel="Sequence length", ylabel="Forward p50 (ms)", yscale="log")
    axis.legend()
    axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(args.assets / "flash_forward_latency.png", dpi=160)
    plt.close(figure)


if __name__ == "__main__":
    main()
