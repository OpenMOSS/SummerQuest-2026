from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "torch_eager": "#4C78A8",
    "torch_compiled": "#F58518",
    "flash_triton": "#54A24B",
}
LABELS = {
    "torch_eager": "PyTorch eager",
    "torch_compiled": "PyTorch compiled",
    "flash_triton": "Triton FlashAttention",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as source:
        return list(csv.DictReader(source))


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 140,
        }
    )


def render_checkpoint(results: Path, assets: Path) -> None:
    rows = [
        row
        for row in read_csv(results / "checkpointing.csv")
        if row["status"] == "success"
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.6))
    for context, color, marker in (
        (1024, "#4C78A8", "o"),
        (2048, "#E45756", "s"),
    ):
        selected = [row for row in rows if int(row["context_length"]) == context]
        axis.scatter(
            [float(row["peak_reserved_mib"]) / 1024 for row in selected],
            [float(row["step_time_ms_p50"]) for row in selected],
            color=color,
            marker=marker,
            s=55,
            label=f"context {context}",
        )
        for row in selected:
            block = row["checkpoint_block_size"]
            label = "none" if block in ("", "None") else f"block {block}"
            normalized_block = None if block in ("", "None") else int(block)
            offsets = {
                (1024, None): (8, 4),
                (1024, 1): (8, 12),
                (1024, 2): (10, 2),
                (1024, 4): (8, -14),
                (1024, 8): (8, 10),
                (2048, None): (8, 5),
                (2048, 1): (8, 7),
            }
            axis.annotate(
                label,
                (
                    float(row["peak_reserved_mib"]) / 1024,
                    float(row["step_time_ms_p50"]),
                ),
                xytext=offsets[(context, normalized_block)],
                textcoords="offset points",
                fontsize=8,
            )
    axis.set_xlabel("Peak reserved memory (GiB)")
    axis.set_ylabel("Training-step p50 latency (ms)")
    axis.set_title("Activation checkpointing: memory/latency trade-off")
    axis.legend()
    figure.tight_layout()
    figure.savefig(
        assets / "checkpoint_tradeoff.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def render_flash_speedup(results: Path, assets: Path) -> None:
    rows = [
        row
        for row in read_csv(results / "flash_benchmark.csv")
        if row["implementation"] == "flash_triton"
        and row["status"] == "success"
    ]
    phases = ("forward", "backward", "forward_backward")
    figure, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), sharey=True)
    for axis, phase in zip(axes, phases, strict=True):
        selected = [row for row in rows if row["phase"] == phase]
        for head_dim, marker in ((64, "o"), (128, "s")):
            points = sorted(
                (
                    int(row["seq_len"]),
                    float(row["speedup_vs_eager"]),
                )
                for row in selected
                if int(row["head_dim"]) == head_dim
            )
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker=marker,
                linewidth=1.8,
                label=f"head dim {head_dim}",
            )
        axis.axhline(1.0, color="#777777", linestyle="--", linewidth=1)
        axis.set_xscale("log", base=2)
        axis.set_xticks([512, 2048, 8192, 16384])
        axis.set_xticklabels(["512", "2K", "8K", "16K"])
        axis.set_title(phase.replace("_", " + "))
        axis.set_xlabel("Sequence length")
    axes[0].set_ylabel("Speedup vs equal-shape eager")
    axes[-1].legend(loc="upper left")
    figure.suptitle("Triton FlashAttention p50 speedup")
    figure.tight_layout()
    figure.savefig(
        assets / "flash_speedup.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def render_forward_memory(results: Path, assets: Path) -> None:
    rows = [
        row
        for row in read_csv(results / "flash_benchmark.csv")
        if row["phase"] == "forward" and row["status"] == "success"
    ]
    figure, axes = plt.subplots(1, 2, figsize=(9.6, 3.9), sharey=True)
    for axis, head_dim in zip(axes, (64, 128), strict=True):
        for implementation in (
            "torch_eager",
            "torch_compiled",
            "flash_triton",
        ):
            points = sorted(
                (
                    int(row["seq_len"]),
                    float(row["peak_reserved_mib"]) / 1024,
                )
                for row in rows
                if int(row["head_dim"]) == head_dim
                and row["implementation"] == implementation
            )
            if not points:
                continue
            axis.plot(
                [point[0] for point in points],
                [point[1] for point in points],
                marker="o",
                color=COLORS[implementation],
                linewidth=1.8,
                label=LABELS[implementation],
            )
        axis.set_xscale("log", base=2)
        axis.set_xticks([512, 2048, 8192, 16384])
        axis.set_xticklabels(["512", "2K", "8K", "16K"])
        axis.set_xlabel("Sequence length")
        axis.set_title(f"head dim {head_dim}")
    axes[0].set_ylabel("Peak reserved memory (GiB)")
    axes[-1].legend()
    figure.suptitle("Forward-pass memory scaling")
    figure.tight_layout()
    figure.savefig(
        assets / "forward_memory.png",
        dpi=180,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    args = parser.parse_args()
    args.assets.mkdir(parents=True, exist_ok=True)
    configure_style()
    render_checkpoint(args.results, args.assets)
    render_flash_speedup(args.results, args.assets)
    render_forward_memory(args.results, args.assets)


if __name__ == "__main__":
    main()
