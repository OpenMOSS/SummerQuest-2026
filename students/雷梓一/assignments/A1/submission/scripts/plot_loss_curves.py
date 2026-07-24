from __future__ import annotations

import argparse
import html
import json
import math
from pathlib import Path


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#be123c"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render JSONL loss curves as a standalone SVG.")
    parser.add_argument("--series", action="append", required=True, help="LABEL=PATH_TO_JSONL")
    parser.add_argument("--x", choices=["step", "wall_clock_sec"], default="step")
    parser.add_argument("--y", choices=["train_loss", "val_loss"], default="val_loss")
    parser.add_argument("--y-min", type=float)
    parser.add_argument("--y-max", type=float)
    parser.add_argument("--title", required=True)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args()


def load_points(path: Path, x_key: str, y_key: str) -> list[tuple[float, float]]:
    points = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            record = json.loads(line)
            if x_key not in record or y_key not in record:
                continue
            x_value = float(record[x_key])
            y_value = float(record[y_key])
            if math.isfinite(x_value) and math.isfinite(y_value):
                points.append((x_value, y_value))
    return points


def main() -> None:
    args = parse_args()
    series: list[tuple[str, list[tuple[float, float]]]] = []
    for specification in args.series:
        label, separator, path = specification.partition("=")
        if not separator:
            raise ValueError(f"invalid --series value: {specification!r}")
        points = load_points(Path(path), args.x, args.y)
        if points:
            series.append((label, points))
    if not series:
        raise ValueError("no finite points found")

    all_x = [x for _, points in series for x, _ in points]
    all_y = [y for _, points in series for _, y in points]
    x_min, x_max = min(all_x), max(all_x)
    y_min = args.y_min if args.y_min is not None else min(all_y)
    y_max = args.y_max if args.y_max is not None else max(all_y)
    if x_max == x_min:
        x_max += 1
    if y_max == y_min:
        y_max += 1

    width, height = 1000, 620
    left, right, top, bottom = 90, 30, 65, 75
    plot_width = width - left - right
    plot_height = height - top - bottom

    def scale_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def scale_y(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return top + (y_max - clipped) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{html.escape(args.title)}</text>',
    ]
    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        x_value = x_min + fraction * (x_max - x_min)
        y = top + fraction * plot_height
        y_value = y_max - fraction * (y_max - y_min)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{x:.1f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.3g}</text>')
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.3g}</text>')
    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="2"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="2"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="15">{html.escape(args.x)}</text>',
            f'<text x="20" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 20 {top + plot_height / 2})" font-family="sans-serif" font-size="15">{html.escape(args.y)}</text>',
        ]
    )
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in points)
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"/>')
        legend_x = left + 12 + (index % 4) * 205
        legend_y = top + 18 + (index // 4) * 22
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(f'<text x="{legend_x + 31}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{html.escape(label)}</text>')
    elements.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(elements) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
