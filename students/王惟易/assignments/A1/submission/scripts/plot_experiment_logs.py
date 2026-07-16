import argparse
import json
import math
from pathlib import Path

import matplotlib.pyplot as plt


COLORS = {
    "blue": "#0072B2",
    "orange": "#E69F00",
    "green": "#009E73",
    "red": "#D55E00",
    "purple": "#CC79A7",
    "sky": "#56B4E9",
    "black": "#222222",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot A1 JSONL experiment logs")
    parser.add_argument("--logs-root", type=Path, default=Path("logs/experiments/raw"))
    parser.add_argument("--output-dir", type=Path, default=Path("assets"))
    return parser.parse_args()


def read_metrics(logs_root: Path, run_name: str) -> list[dict[str, float]]:
    path = logs_root / run_name / "metrics.jsonl"
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def finite_points(
    rows: list[dict[str, float]],
    key: str,
    x_key: str = "step",
) -> tuple[list[float], list[float]]:
    points = [
        (float(row[x_key]), float(row[key]))
        for row in rows
        if key in row and math.isfinite(float(row[key]))
    ]
    return [point[0] for point in points], [point[1] for point in points]


def configure_style() -> None:
    plt.style.use("seaborn-v0_8-whitegrid")
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 10,
            "legend.fontsize": 9,
            "svg.fonttype": "none",
        }
    )


def save_figure(figure: plt.Figure, output_path: Path) -> None:
    figure.savefig(output_path, format="svg", bbox_inches="tight")
    plt.close(figure)


def plot_baseline(logs_root: Path, output_dir: Path) -> None:
    rows = read_metrics(logs_root, "tinystories_baseline_full")
    train_steps, train_losses = finite_points(rows, "train_loss")
    val_steps, val_losses = finite_points(rows, "val_loss")
    val_times, val_losses_by_time = finite_points(rows, "val_loss", "wall_clock_sec")

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(train_steps, train_losses, color=COLORS["sky"], alpha=0.55, linewidth=1, label="Train")
    axes[0].plot(val_steps, val_losses, color=COLORS["blue"], linewidth=2, label="Validation")
    axes[0].set(title="TinyStories baseline", xlabel="Optimizer step", ylabel="Cross-entropy loss")
    axes[0].legend()

    axes[1].plot(
        [seconds / 60 for seconds in val_times],
        val_losses_by_time,
        color=COLORS["blue"],
        linewidth=2,
    )
    axes[1].set(title="Validation loss by wall clock", xlabel="Wall-clock minutes", ylabel="Validation loss")
    figure.suptitle("TinyStories full baseline: 327.68M training tokens")
    save_figure(figure, output_dir / "tinystories_baseline_loss.svg")


def plot_learning_rates(logs_root: Path, output_dir: Path) -> None:
    groups = [
        [
            ("tinystories_lr_3e-4", "3e-4", COLORS["sky"]),
            ("tinystories_lr_1e-3", "1e-3", COLORS["green"]),
            ("tinystories_lr_3e-3", "3e-3", COLORS["blue"]),
            ("tinystories_lr_1e-2", "1e-2", COLORS["orange"]),
        ],
        [
            ("tinystories_lr_3e-2", "3e-2", COLORS["purple"]),
            ("tinystories_lr_1e-1", "1e-1", COLORS["red"]),
        ],
    ]

    figure, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for axis, runs in zip(axes, groups, strict=True):
        for run_name, label, color in runs:
            rows = read_metrics(logs_root, run_name)
            steps, losses = finite_points(rows, "val_loss")
            processed_millions = [step * 64 * 256 / 1e6 for step in steps]
            axis.plot(processed_millions, losses, marker="o", markersize=3, linewidth=1.8, label=label, color=color)
        axis.set(xlabel="Processed tokens (millions)", ylabel="Validation loss")
        axis.legend(title="Max LR")

    axes[0].set_title("Stable search region")
    axes[1].set_title("Beyond the stability edge")
    figure.suptitle("TinyStories learning-rate sweep")
    save_figure(figure, output_dir / "tinystories_learning_rate_sweep.svg")


def plot_batch_sizes(logs_root: Path, output_dir: Path) -> None:
    runs = [
        ("tinystories_batch_1", "Batch 1", 1, COLORS["black"]),
        ("tinystories_batch_16", "Batch 16", 16, COLORS["green"]),
        ("tinystories_batch_32", "Batch 32", 32, COLORS["blue"]),
        ("tinystories_batch_64", "Batch 64", 64, COLORS["orange"]),
    ]

    figure, axis = plt.subplots(figsize=(6.8, 4.5))
    for run_name, label, batch_size, color in runs:
        rows = read_metrics(logs_root, run_name)
        steps, losses = finite_points(rows, "val_loss")
        processed_millions = [step * batch_size * 256 / 1e6 for step in steps]
        axis.plot(processed_millions, losses, marker="o", linewidth=1.8, label=label, color=color)

    axis.set(
        title="Batch-size sweep at a fixed 2.10M-token budget",
        xlabel="Processed tokens (millions)",
        ylabel="Validation loss",
    )
    axis.legend()
    save_figure(figure, output_dir / "tinystories_batch_size_sweep.svg")


def plot_ablations(logs_root: Path, output_dir: Path) -> None:
    runs = [
        ("tinystories_ablation_control", "Control", COLORS["black"]),
        ("tinystories_ablation_post_norm", "Post-Norm", COLORS["green"]),
        ("tinystories_ablation_nope", "NoPE", COLORS["orange"]),
        ("tinystories_ablation_silu", "SiLU FFN", COLORS["blue"]),
        ("tinystories_ablation_no_norm_low_lr", "No RMSNorm, LR 1e-3", COLORS["purple"]),
    ]

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    for run_name, label, color in runs:
        rows = read_metrics(logs_root, run_name)
        steps, losses = finite_points(rows, "val_loss")
        axis.plot(steps, losses, marker="o", markersize=3, linewidth=1.8, label=label, color=color)

    unstable_rows = read_metrics(logs_root, "tinystories_ablation_no_norm")
    unstable_steps, unstable_losses = finite_points(unstable_rows, "val_loss")
    axis.scatter(unstable_steps, unstable_losses, marker="x", s=55, color=COLORS["red"], label="No RMSNorm, LR 3e-3")
    axis.axvline(180, color=COLORS["red"], linestyle="--", alpha=0.7, linewidth=1)
    axis.annotate("first NaN at step 180", xy=(180, 3.15), xytext=(290, 3.35), arrowprops={"arrowstyle": "->", "color": COLORS["red"]}, color=COLORS["red"])

    axis.set(
        title="TinyStories architecture ablations: 32.77M tokens per run",
        xlabel="Optimizer step",
        ylabel="Validation loss",
    )
    axis.legend(ncol=2)
    save_figure(figure, output_dir / "tinystories_ablations.svg")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    configure_style()
    plot_baseline(args.logs_root, args.output_dir)
    plot_learning_rates(args.logs_root, args.output_dir)
    plot_batch_sizes(args.logs_root, args.output_dir)
    plot_ablations(args.logs_root, args.output_dir)


if __name__ == "__main__":
    main()
