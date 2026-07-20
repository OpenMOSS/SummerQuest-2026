"""根据 JSONL 日志绘制 RoPE 与 NoPE validation-loss 对比曲线。"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_validation_points(path: Path) -> list[tuple[float, float]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    points = [
        (float(row["processed_tokens"]) / 1e6, float(row["val_loss"]))
        for row in rows
        if "val_loss" in row
    ]
    if not points:
        raise ValueError(f"{path} does not contain validation-loss points")
    return points


def write_svg(rope_path: Path, nope_path: Path, output_path: Path) -> None:
    series = [("RoPE", "#2563eb", load_validation_points(rope_path)), ("NoPE", "#dc2626", load_validation_points(nope_path))]
    width, height = 960, 570
    left, right, top, bottom = 88, 40, 72, 78
    plot_width, plot_height = width - left - right, height - top - bottom
    all_points = [point for _, _, points in series for point in points]
    maximum_tokens = max(tokens for tokens, _ in all_points)
    lower, upper = min(loss for _, loss in all_points), max(loss for _, loss in all_points)
    margin = max((upper - lower) * 0.08, 0.03)
    lower, upper = lower - margin, upper + margin

    def x(tokens: float) -> float:
        return left + tokens / maximum_tokens * plot_width

    def y(loss: float) -> float:
        return top + (upper - loss) / (upper - lower) * plot_height

    def polyline(points: list[tuple[float, float]]) -> str:
        return " ".join(f"{x(tokens):.2f},{y(loss):.2f}" for tokens, loss in points)

    svg = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>text{font-family:sans-serif;fill:#1f2937}.tick{font-size:13px}</style>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-size="20" font-weight="bold">TinyStories position-encoding ablation</text>',
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
    for index, (label, color, points) in enumerate(series):
        legend_x = left + 16 + index * 118
        svg += [
            f'<polyline points="{polyline(points)}" fill="none" stroke="{color}" stroke-width="3"/>',
            f'<line x1="{legend_x}" y1="{top + 8}" x2="{legend_x + 24}" y2="{top + 8}" stroke="{color}" stroke-width="3"/>',
            f'<text class="tick" x="{legend_x + 30}" y="{top + 13}">{label}</text>',
        ]
    svg += [
        f'<text x="{width / 2}" y="{height - 20}" text-anchor="middle" font-size="15">processed training tokens</text>',
        f'<text x="22" y="{height / 2}" text-anchor="middle" font-size="15" transform="rotate(-90 22 {height / 2})">validation cross-entropy loss (per token)</text>',
        "</svg>",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(svg), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rope", type=Path, required=True, help="RoPE baseline metrics.jsonl")
    parser.add_argument("--nope", type=Path, required=True, help="NoPE metrics.jsonl")
    parser.add_argument("--output", type=Path, default=Path("assets/tinystories_rope_vs_nope.svg"))
    args = parser.parse_args()
    write_svg(args.rope, args.nope, args.output)
    print(f"曲线已写入 {args.output}")


if __name__ == "__main__":
    main()
