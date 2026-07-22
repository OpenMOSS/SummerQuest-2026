"""读取 batch sweep 日志，绘制不同每卡 batch 的 validation-loss 学习曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from omegaconf import OmegaConf


ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "submission":
    ROOT = ROOT.parent
DEFAULT_LOG_ROOT = ROOT / "logs" / "batch_sweep"
DEFAULT_OUTPUT = ROOT / "assets" / "tinystories_batch_sweep.svg"
PALETTE = ("#2563eb", "#16a34a", "#dc2626", "#9333ea", "#ea580c", "#0891b2", "#be123c")


def load_runs(log_root: Path) -> list[dict[str, object]]:
    """读入每个完成 run 的每卡 batch、学习率和 validation 曲线。"""
    runs = []
    for metrics_path in sorted(log_root.glob("**/metrics.jsonl")):
        run_directory = metrics_path.parent
        config_path = run_directory / "config.yaml"
        if not config_path.exists():
            continue
        config = OmegaConf.load(config_path)
        rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
        points = [
            (float(row["processed_tokens"]) / 1e6, float(row["val_loss"])) for row in rows if "val_loss" in row
        ]
        if points:
            runs.append(
                {
                    "batch_size": int(config.training.batch_size),
                    "peak_lr": float(config.optimizer.max_lr),
                    "points": points,
                }
            )
    return sorted(runs, key=lambda run: int(run["batch_size"]))


def write_svg(runs: list[dict[str, object]], output_path: Path) -> None:
    """不用额外绘图库生成可嵌入 Markdown 的 SVG。"""
    if not runs:
        raise ValueError("没有找到包含 validation loss 的 batch sweep 日志")
    width, height = 980, 580
    left, right, top, bottom = 88, 42, 74, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    all_points = [point for run in runs for point in run["points"]]  # type: ignore[misc]
    maximum_tokens = max(tokens for tokens, _ in all_points)
    losses = [loss for _, loss in all_points]
    lower, upper = min(losses), max(losses)
    margin = max((upper - lower) * 0.08, 0.03)
    lower, upper = lower - margin, upper + margin

    def x(tokens_millions: float) -> float:
        return left + tokens_millions / maximum_tokens * plot_width

    def y(loss: float) -> float:
        return top + (upper - loss) / (upper - lower) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#1f2937}.tick{font-size:13px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">TinyStories batch-size sweep</text>',
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#374151"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#374151"/>',
    ]
    for fraction in range(6):
        loss = lower + (upper - lower) * fraction / 5
        coordinate = y(loss)
        svg += [
            f'<line x1="{left}" y1="{coordinate:.2f}" x2="{left + plot_width}" y2="{coordinate:.2f}" stroke="#e5e7eb"/>',
            f'<text class="tick" x="{left - 10}" y="{coordinate + 4:.2f}" text-anchor="end">{loss:.2f}</text>',
        ]
    for fraction in range(6):
        tokens = maximum_tokens * fraction / 5
        coordinate = x(tokens)
        svg += [
            f'<line x1="{coordinate:.2f}" y1="{top + plot_height}" x2="{coordinate:.2f}" y2="{top + plot_height + 5}" stroke="#374151"/>',
            f'<text class="tick" x="{coordinate:.2f}" y="{top + plot_height + 26}" text-anchor="middle">{tokens:.0f}M</text>',
        ]
    for index, run in enumerate(runs):
        color = PALETTE[index % len(PALETTE)]
        points = run["points"]
        point_text = " ".join(f"{x(tokens):.2f},{y(loss):.2f}" for tokens, loss in points)  # type: ignore[misc]
        legend_y = top + 20 * index
        svg += [
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.8"/>',
            f'<line x1="{left + 12}" y1="{legend_y - 5}" x2="{left + 36}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>',
            f'<text class="tick" x="{left + 42}" y="{legend_y}">per-GPU batch={run["batch_size"]}, peak LR={run["peak_lr"]:.2g}</text>',
        ]
    svg += [
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">processed training tokens</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">validation loss (per token)</text>',
        "</svg>",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log-root", type=Path, default=DEFAULT_LOG_ROOT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    runs = load_runs(args.log_root)
    write_svg(runs, args.output)
    print("per_gpu_batch  peak_lr    final_val_loss")
    for run in runs:
        final_loss = run["points"][-1][1]  # type: ignore[index]
        print(f"{run['batch_size']:<14} {run['peak_lr']:<10.3g} {final_loss:.6f}")
    print(f"\n曲线已写入 {args.output}")


if __name__ == "__main__":
    main()
