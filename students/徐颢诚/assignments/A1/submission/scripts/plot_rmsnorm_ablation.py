"""绘制 baseline 与移除 RMSNorm 后不同学习率的 validation-loss 曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if ROOT.name == "submission":
    ROOT = ROOT.parent
DEFAULT_OUTPUT = ROOT / "assets" / "tinystories_rmsnorm_ablation.svg"
DEFAULT_RUNS = (
    (
        "RMSNorm, peak LR=1e-3",
        ROOT / "logs" / "ts_baseline" / "2026-07-15" / "08-16-44" / "metrics.jsonl",
        "#2563eb",
    ),
    (
        "No RMSNorm, peak LR=1e-3",
        ROOT / "logs" / "ablation_no_rmsnorm_lr1e3" / "2026-07-15" / "10-43-56" / "metrics.jsonl",
        "#dc2626",
    ),
    (
        "No RMSNorm, peak LR=3e-4",
        ROOT / "logs" / "ablation_no_rmsnorm_lr3e4" / "2026-07-15" / "10-56-37" / "metrics.jsonl",
        "#16a34a",
    ),
)


def read_validation_points(path: Path) -> list[tuple[float, float]]:
    """以百万 processed tokens 为横轴读取 validation loss。"""
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    return [(float(row["processed_tokens"]) / 1e6, float(row["val_loss"])) for row in rows if "val_loss" in row]


def write_svg(output_path: Path) -> None:
    """使用标准库生成可直接嵌入 Homework.md 的 SVG。"""
    runs = [(label, read_validation_points(path), color) for label, path, color in DEFAULT_RUNS]
    if any(not points for _, points, _ in runs):
        raise ValueError("每个 run 都必须至少包含一个 validation-loss 点")

    width, height = 980, 580
    left, right, top, bottom = 88, 42, 74, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    all_points = [point for _, points, _ in runs for point in points]
    maximum_tokens = max(tokens for tokens, _ in all_points)
    losses = [loss for _, loss in all_points]
    lower, upper = min(losses), max(losses)
    margin = max((upper - lower) * 0.08, 0.03)
    lower, upper = lower - margin, upper + margin

    def x(tokens: float) -> float:
        return left + tokens / maximum_tokens * plot_width

    def y(loss: float) -> float:
        return top + (upper - loss) / (upper - lower) * plot_height

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#1f2937}.tick{font-size:13px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">RMSNorm ablation on TinyStories</text>',
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
    for index, (label, points, color) in enumerate(runs):
        point_text = " ".join(f"{x(tokens):.2f},{y(loss):.2f}" for tokens, loss in points)
        legend_y = top + index * 20
        svg += [
            f'<polyline points="{point_text}" fill="none" stroke="{color}" stroke-width="2.8"/>',
            f'<line x1="{left + 12}" y1="{legend_y - 5}" x2="{left + 36}" y2="{legend_y - 5}" stroke="{color}" stroke-width="3"/>',
            f'<text class="tick" x="{left + 42}" y="{legend_y}">{label}</text>',
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
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    write_svg(args.output)
    print(f"曲线已写入 {args.output}")


if __name__ == "__main__":
    main()
