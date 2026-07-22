#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt


def read_metrics(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def draw_group(runs: list[tuple[str, Path]], title: str, output: Path, x_key: str = "step") -> None:
    fig, axis = plt.subplots(figsize=(7.2, 4.2))
    for label, path in runs:
        records = read_metrics(path)
        validation = [record for record in records if "val_loss" in record]
        if validation:
            axis.plot(
                [record[x_key] for record in validation],
                [record["val_loss"] for record in validation],
                marker="o",
                markersize=2.5,
                linewidth=1.4,
                label=label,
            )
    axis.set_title(title)
    axis.set_xlabel("Step" if x_key == "step" else "Wall-clock time (s)")
    axis.set_ylabel("Validation loss")
    axis.grid(alpha=0.25)
    axis.legend(fontsize=8)
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot A1 validation-loss curves from JSONL logs.")
    parser.add_argument("--artifacts", type=Path, default=Path("artifacts"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    experiments = args.artifacts / "experiments"
    args.output_dir.mkdir(parents=True, exist_ok=True)

    draw_group(
        [(path.parent.name.removeprefix("lr_"), path) for path in sorted(experiments.glob("lr_*/metrics.jsonl"))],
        "TinyStories learning-rate sweep",
        args.output_dir / "learning_rate.svg",
    )
    draw_group(
        [(path.parent.name.removeprefix("batch_"), path) for path in sorted(experiments.glob("batch_*/metrics.jsonl"))],
        "TinyStories batch-size comparison",
        args.output_dir / "batch_size.svg",
    )
    draw_group(
        [
            (path.parent.name.removeprefix("ablation_"), path)
            for path in sorted(experiments.glob("ablation_*/metrics.jsonl"))
        ],
        "TinyStories architecture ablations",
        args.output_dir / "ablations.svg",
    )
    main_runs = [
        ("TinyStories", args.artifacts / "runs/tinystories_baseline/metrics.jsonl"),
        ("OpenWebText", args.artifacts / "runs/owt_baseline/metrics.jsonl"),
    ]
    draw_group(main_runs, "Main training runs", args.output_dir / "main_training.svg")
    draw_group(main_runs, "Main training runs by wall-clock time", args.output_dir / "main_training_time.svg", "wall_clock_sec")


if __name__ == "__main__":
    main()
