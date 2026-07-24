"""从一次训练的 JSONL 日志生成 train/validation loss 曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "submission":
    ROOT = ROOT.parent
DEFAULT_METRICS = ROOT / "logs" / "ts_baseline" / "2026-07-15" / "08-16-44" / "metrics.jsonl"
DEFAULT_OUTPUT = ROOT / "assets" / "tinystories_baseline_loss.svg"


def write_svg(metrics_path: Path, output_path: Path) -> None:
    """以 processed tokens 为横轴，绘制训练与验证交叉熵。"""
    rows = [json.loads(line) for line in metrics_path.read_text(encoding="utf-8").splitlines() if line]
    train_points = [(float(row["processed_tokens"]) / 1e6, float(row["train_loss"])) for row in rows]
    validation_points = [
        (float(row["processed_tokens"]) / 1e6, float(row["val_loss"])) for row in rows if "val_loss" in row
    ]
    if not train_points or not validation_points:
        raise ValueError("metrics.jsonl must contain both train_loss and val_loss points")

    width, height = 960, 570
    left, right, top, bottom = 88, 40, 72, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    all_losses = [loss for _, loss in train_points + validation_points]
    lower, upper = min(all_losses), max(all_losses)
    margin = max((upper - lower) * 0.08, 0.03)
    lower, upper = lower - margin, upper + margin
    maximum_tokens = max(tokens for tokens, _ in train_points)

    def x(tokens_millions: float) -> float:
        return left + tokens_millions / maximum_tokens * plot_width

    def y(loss: float) -> float:
        return top + (upper - loss) / (upper - lower) * plot_height

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x(tokens):.2f},{y(loss):.2f}" for tokens, loss in points)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#1f2937}.tick{font-size:13px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">TinyStories baseline training curves</text>',
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
    target_y = y(1.45)
    if top <= target_y <= top + plot_height:
        svg += [
            f'<line x1="{left}" y1="{target_y:.2f}" x2="{left + plot_width}" y2="{target_y:.2f}" stroke="#111827" stroke-dasharray="6 4"/>',
            f'<text class="tick" x="{left + plot_width - 4}" y="{target_y - 6:.2f}" text-anchor="end">target: 1.45</text>',
        ]
    svg += [
        f'<polyline points="{polyline(train_points)}" fill="none" stroke="#2563eb" stroke-width="2.5"/>',
        f'<polyline points="{polyline(validation_points)}" fill="none" stroke="#dc2626" stroke-width="3"/>',
        f'<line x1="{left + 16}" y1="{top + 8}" x2="{left + 40}" y2="{top + 8}" stroke="#2563eb" stroke-width="3"/>',
        f'<text class="tick" x="{left + 46}" y="{top + 13}">train loss</text>',
        f'<line x1="{left + 126}" y1="{top + 8}" x2="{left + 150}" y2="{top + 8}" stroke="#dc2626" stroke-width="3"/>',
        f'<text class="tick" x="{left + 156}" y="{top + 13}">validation loss</text>',
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">processed training tokens</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">cross-entropy loss (per token)</text>',
        "</svg>",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--metrics", type=Path, default=DEFAULT_METRICS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_svg(args.metrics, args.output)
    print(f"曲线已写入 {args.output}")


if __name__ == "__main__":
    main()
