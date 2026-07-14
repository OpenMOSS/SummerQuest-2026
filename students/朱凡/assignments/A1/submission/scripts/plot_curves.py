"""Plot section 7 validation-loss curves from aggregated run logs."""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parent.parent
SUBMISSION_LOGS_DIR = REPO_ROOT / "data" / "submission_logs"
ASSETS_DIR = REPO_ROOT / "assets"
VALID_X_FIELDS = {"iteration", "tokens_seen", "process_elapsed_seconds", "training_elapsed_seconds"}


def _validation_points(path: Path, x_field: str, max_x: float | None) -> list[tuple[float, float]]:
    points = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            if "validation_loss" not in record or x_field not in record:
                continue
            x_value = float(record[x_field])
            loss = float(record["validation_loss"])
            if max_x is None or x_value <= max_x:
                points.append((x_value, loss))
    return points


def plot_group(
    run_names: Iterable[str],
    title: str,
    out_path: Path | str,
    *,
    x_field: str = "tokens_seen",
    max_x: float | None = None,
    require_points: bool = False,
    x_scale: float = 1.0,
    x_label: str | None = None,
) -> None:
    if x_field not in VALID_X_FIELDS:
        raise ValueError(f"unsupported x field: {x_field}")
    fig, axis = plt.subplots(figsize=(8, 5))
    plotted = 0
    for name in run_names:
        path = SUBMISSION_LOGS_DIR / name / "metrics.jsonl"
        if not path.exists():
            continue
        points = _validation_points(path, x_field, max_x)
        if not points:
            continue
        xs, ys = zip(*points)
        scaled_xs = [value / x_scale for value in xs]
        axis.plot(scaled_xs, ys, label=name, marker="o", markersize=3)
        plotted += 1
        axis.annotate(f"{ys[-1]:.3f}", (scaled_xs[-1], ys[-1]), fontsize=7)
    if require_points and plotted == 0:
        plt.close(fig)
        raise ValueError(f"no valid points for required plot {title!r}")
    axis.set_xlabel(x_label or x_field)
    axis.set_ylabel("validation loss")
    axis.set_title(title)
    axis.grid(True, alpha=0.3)
    if plotted:
        axis.legend()
    output = Path(out_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    lr_runs = ["lr_1e-4", "lr_3e-4", "lr_1e-3", "lr_3e-3", "lr_1e-2"]
    batch_runs = sorted(path.name for path in SUBMISSION_LOGS_DIR.glob("batch_*") if path.is_dir())
    ablation_runs = [
        "tinystories_baseline",
        "ablation_no_rmsnorm",
        "ablation_no_rmsnorm_low_lr",
        "ablation_post_norm",
        "ablation_nope",
        "ablation_silu",
    ]
    plot_group(lr_runs, "Learning-rate sweep", ASSETS_DIR / "lr_sweep_tokens.png")
    plot_group(batch_runs, "Batch size by tokens", ASSETS_DIR / "batch_size_tokens.png")
    plot_group(
        batch_runs,
        "Batch size by wall clock",
        ASSETS_DIR / "batch_size_wallclock.png",
        x_field="training_elapsed_seconds",
        x_scale=60,
        x_label="training time (minutes)",
    )
    plot_group(ablation_runs, "Architecture ablations", ASSETS_DIR / "ablations_tokens.png")
    plot_group(
        ["tinystories_baseline", "owt_baseline"],
        "TinyStories vs OpenWebText",
        ASSETS_DIR / "ts_vs_owt_iterations.png",
        x_field="iteration",
    )
    plot_group(
        ["leaderboard"],
        "Leaderboard (<45 minutes)",
        ASSETS_DIR / "leaderboard_wallclock.png",
        x_field="process_elapsed_seconds",
        max_x=2700,
        require_points=True,
        x_scale=60,
        x_label="process wall time (minutes)",
    )


if __name__ == "__main__":
    main()
