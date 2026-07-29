"""Render compact SVG figures directly from the submitted CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as pyplot


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint-csv", type=Path, required=True)
    parser.add_argument("--flash-csv", type=Path, required=True)
    parser.add_argument("--output-directory", type=Path, required=True)
    return parser.parse_args()


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_figure(rows: list[dict[str, str]], output: Path) -> None:
    rows = [row for row in rows if row["context_length"] == "1024" and row["status"] == "ok"]
    labels = ["none" if row["checkpoint_block_size"] == "none" else f"B={row['checkpoint_block_size']}" for row in rows]
    latencies = [float(row["step_time_ms_p50"]) for row in rows]
    memory = [float(row["peak_allocated_mib"]) for row in rows]
    figure, latency_axis = pyplot.subplots(figsize=(7.2, 4.2))
    memory_axis = latency_axis.twinx()
    positions = list(range(len(rows)))
    latency_axis.plot(positions, latencies, "o-", color="#2563eb", label="step p50")
    memory_axis.plot(
        positions,
        memory,
        "s--",
        color="#dc2626",
        label="peak allocated",
    )
    latency_axis.set_xticks(positions, labels)
    latency_axis.set_ylabel("Step p50 (ms)", color="#2563eb")
    memory_axis.set_ylabel("Peak allocated (MiB)", color="#dc2626")
    latency_axis.set_title("Checkpoint recomputation–memory trade-off (ctx=1024)")
    latency_axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def attention_figure(rows: list[dict[str, str]], output: Path) -> None:
    selected = [row for row in rows if row["head_dim"] == "128" and row["phase"] == "forward" and row["status"] == "ok"]
    figure, axis = pyplot.subplots(figsize=(7.2, 4.2))
    for implementation, color in (
        ("eager", "#dc2626"),
        ("compiled", "#16a34a"),
        ("triton", "#2563eb"),
    ):
        implementation_rows = sorted(
            (row for row in selected if row["implementation"] == implementation),
            key=lambda row: int(row["sequence_length"]),
        )
        if not implementation_rows:
            continue
        axis.plot(
            [int(row["sequence_length"]) for row in implementation_rows],
            [float(row["p50_ms"]) for row in implementation_rows],
            "o-",
            label=implementation,
            color=color,
        )
    axis.set_xscale("log", base=2)
    axis.set_yscale("log")
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Forward p50 (ms, log scale)")
    axis.set_title("Causal BF16 attention, head dimension 128")
    axis.grid(which="both", alpha=0.25)
    axis.legend()
    figure.tight_layout()
    figure.savefig(output, format="svg")
    pyplot.close(figure)


def main() -> int:
    args = parse_args()
    args.output_directory.mkdir(parents=True, exist_ok=True)
    checkpoint_figure(
        read_rows(args.checkpoint_csv),
        args.output_directory / "checkpoint_tradeoff.svg",
    )
    attention_figure(
        read_rows(args.flash_csv),
        args.output_directory / "attention_latency.svg",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
