from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


def parser() -> argparse.Namespace:
    result = argparse.ArgumentParser(description="Generate the two required A2-K report figures")
    result.add_argument("--results-dir", type=Path, default=Path("results"))
    result.add_argument("--output-dir", type=Path, default=Path("assets"))
    return result.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def checkpoint_plot(results_dir: Path, output_dir: Path) -> Path:
    rows = [
        row
        for row in read_csv(results_dir / "checkpointing.csv")
        if row["context_length"] == "1024" and row["status"] == "success"
    ]
    rows.sort(key=lambda row: -1 if not row["checkpoint_block_size"] else int(row["checkpoint_block_size"]))
    if not rows:
        raise ValueError("checkpointing.csv has no successful context-1024 rows")
    labels = ["none" if not row["checkpoint_block_size"] else row["checkpoint_block_size"] for row in rows]
    memory = [float(row["peak_allocated_mib"]) / 1024 for row in rows]
    latency = [float(row["step_time_ms_p50"]) for row in rows]

    figure, memory_axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    time_axis = memory_axis.twinx()
    positions = range(len(rows))
    memory_axis.bar(positions, memory, color="#4C78A8", alpha=0.82, label="Peak allocated")
    time_axis.plot(positions, latency, color="#E45756", marker="o", linewidth=2, label="Step p50")
    memory_axis.set_xticks(list(positions), labels)
    memory_axis.set_xlabel("Checkpoint block size (layers)")
    memory_axis.set_ylabel("Peak allocated (GiB)", color="#4C78A8")
    time_axis.set_ylabel("Training-step p50 (ms)", color="#E45756")
    memory_axis.set_title("Activation checkpointing: memory-compute trade-off")
    memory_axis.grid(axis="y", alpha=0.25)
    output = output_dir / "checkpoint_tradeoff.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def flash_plot(results_dir: Path, output_dir: Path) -> Path:
    rows = [
        row
        for row in read_csv(results_dir / "flash_benchmark.csv")
        if row["phase"] == "forward_backward" and row["dtype"] == "bf16" and row["status"] == "success"
    ]
    if not rows:
        raise ValueError("flash_benchmark.csv has no successful BF16 forward_backward rows")
    figure, axes = plt.subplots(1, 2, figsize=(10.2, 4.2), sharey=True, constrained_layout=True)
    colors = {"eager": "#4C78A8", "compiled": "#F2CF5B", "triton": "#59A14F"}
    for axis, head_dim in zip(axes, (64, 128), strict=True):
        for implementation in ("eager", "compiled", "triton"):
            selected = sorted(
                (row for row in rows if row["head_dim"] == str(head_dim) and row["implementation"] == implementation),
                key=lambda row: int(row["sequence_length"]),
            )
            if not selected:
                continue
            axis.plot(
                [int(row["sequence_length"]) for row in selected],
                [float(row["p50_ms"]) for row in selected],
                marker="o",
                linewidth=2,
                label=implementation,
                color=colors[implementation],
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_title(f"head_dim={head_dim}")
        axis.set_xlabel("Sequence length")
        axis.grid(True, which="both", alpha=0.25)
    axes[0].set_ylabel("Forward-backward p50 (ms)")
    axes[1].legend(title="Implementation")
    figure.suptitle("Causal BF16 attention latency on RTX 4090")
    output = output_dir / "flash_latency.png"
    figure.savefig(output, dpi=180)
    plt.close(figure)
    return output


def main() -> int:
    args = parser()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outputs = [checkpoint_plot(args.results_dir, args.output_dir), flash_plot(args.results_dir, args.output_dir)]
    for output in outputs:
        print(f"wrote {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
