"""Render two compact, headless figures from the public A2-K CSV files."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as pyplot  # noqa: E402


COLORS = {
    "eager": "#526D82",
    "compiled": "#D98E04",
    "triton": "#1B998B",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def checkpoint_figure(
    rows: list[dict[str, str]],
    destination: Path,
) -> None:
    selected = [
        row
        for row in rows
        if row["context_length"] == "1024" and row["status"] == "ok"
    ]
    order = {"none": 0, "1": 1, "2": 2, "4": 3, "8": 4}
    selected.sort(key=lambda row: order[row["checkpoint_block_size"]])
    labels = [
        "none" if row["checkpoint_block_size"] == "none" else row[
            "checkpoint_block_size"
        ]
        for row in selected
    ]
    memory = [float(row["peak_allocated_mib"]) for row in selected]
    latency = [float(row["step_time_ms_p50"]) for row in selected]

    figure, memory_axis = pyplot.subplots(figsize=(8.0, 4.6))
    positions = list(range(len(labels)))
    bars = memory_axis.bar(
        positions,
        memory,
        color="#6C8EBF",
        width=0.62,
        label="Peak allocated",
    )
    memory_axis.set_ylabel("Peak allocated memory (MiB)")
    memory_axis.set_xlabel("Checkpoint block size (layers)")
    memory_axis.set_xticks(positions, labels)
    memory_axis.set_ylim(0, max(memory) * 1.22)
    memory_axis.grid(axis="y", alpha=0.22)
    for bar, value in zip(bars, memory, strict=True):
        memory_axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + max(memory) * 0.025,
            f"{value:,.0f}",
            ha="center",
            va="bottom",
            fontsize=8,
        )

    latency_axis = memory_axis.twinx()
    latency_axis.plot(
        positions,
        latency,
        color="#C43D3D",
        marker="o",
        linewidth=2.0,
        label="Training-step p50",
    )
    latency_axis.set_ylabel("Training-step p50 (ms)")
    latency_axis.set_ylim(0, max(latency) * 1.28)
    for position, value in zip(positions, latency, strict=True):
        latency_axis.text(
            position,
            value + max(latency) * 0.035,
            f"{value:.1f}",
            ha="center",
            va="bottom",
            color="#9E2F2F",
            fontsize=8,
        )

    figure.suptitle(
        "Checkpointing trades activation memory for recomputation "
        "(Medium, context 1024)"
    )
    handles_left, labels_left = memory_axis.get_legend_handles_labels()
    handles_right, labels_right = latency_axis.get_legend_handles_labels()
    memory_axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="upper center",
        ncols=2,
        frameon=False,
    )
    figure.tight_layout()
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        pil_kwargs={"quality": 88, "method": 6},
    )
    pyplot.close(figure)


def flash_forward_figure(
    rows: list[dict[str, str]],
    destination: Path,
) -> None:
    forward_rows = [
        row
        for row in rows
        if row["phase"] == "forward" and row["status"] == "ok"
    ]
    figure, axes = pyplot.subplots(
        1,
        2,
        figsize=(10.8, 5.0),
        sharey=True,
    )
    for axis, head_dim in zip(axes, ("64", "128"), strict=True):
        for implementation in ("eager", "compiled", "triton"):
            selected = [
                row
                for row in forward_rows
                if row["head_dim"] == head_dim
                and row["implementation"] == implementation
            ]
            selected.sort(key=lambda row: int(row["seq_len"]))
            if not selected:
                continue
            axis.plot(
                [int(row["seq_len"]) for row in selected],
                [float(row["p50_ms"]) for row in selected],
                color=COLORS[implementation],
                marker="o",
                linewidth=2.0,
                label=implementation,
            )
        axis.set_xscale("log", base=2)
        axis.set_yscale("log")
        axis.set_xticks(
            [512, 2048, 8192, 16384],
            ["512", "2K", "8K", "16K"],
        )
        axis.set_title(f"head dimension {head_dim}")
        axis.set_xlabel("Sequence length")
        axis.grid(True, which="both", alpha=0.22)
    axes[0].set_ylabel("Forward p50 latency (ms, log scale)")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.91),
        ncols=3,
        frameon=False,
    )
    figure.suptitle(
        "Explicit eager vs torch.compile vs student Triton forward",
        y=0.985,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.80))
    figure.savefig(
        destination,
        dpi=150,
        bbox_inches="tight",
        pil_kwargs={"quality": 88, "method": 6},
    )
    pyplot.close(figure)


def main() -> int:
    args = parse_args()
    args.assets_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_figure(
        read_csv(args.results_dir / "checkpointing.csv"),
        args.assets_dir / "checkpoint_tradeoff.webp",
    )
    flash_forward_figure(
        read_csv(args.results_dir / "flash_benchmark.csv"),
        args.assets_dir / "flash_forward_latency.webp",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
