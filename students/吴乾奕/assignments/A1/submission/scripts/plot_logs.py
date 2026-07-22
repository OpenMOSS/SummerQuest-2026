#!/usr/bin/env python3
"""Render one metric from JSONL experiment logs as a standalone SVG curve."""

from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path


COLORS = ["#2563eb", "#dc2626", "#059669", "#7c3aed", "#ea580c", "#0891b2", "#4f46e5"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--log", required=True, action="append", type=Path)
    parser.add_argument("--metric", default="validation_loss")
    parser.add_argument("--x", choices=["step", "processed_tokens", "elapsed_seconds"], default="step")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--title", default="A1 experiment curve")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=640)
    return parser.parse_args()


def read_series(path: Path, x_key: str, metric: str) -> tuple[str, list[tuple[float, float]]]:
    points: list[tuple[float, float]] = []
    run_name = path.parent.name
    with path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("run_name"):
                run_name = str(record["run_name"])
            if x_key in record and metric in record:
                x_value = float(record[x_key])
                y_value = float(record[metric])
                if math.isfinite(x_value) and math.isfinite(y_value):
                    points.append((x_value, y_value))
    return run_name, points


def ticks(low: float, high: float, count: int = 5) -> list[float]:
    if low == high:
        return [low]
    return [low + index * (high - low) / count for index in range(count + 1)]


def format_number(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1e9:
        return f"{value / 1e9:.2g}B"
    if magnitude >= 1e6:
        return f"{value / 1e6:.2g}M"
    if magnitude >= 1e3:
        return f"{value / 1e3:.2g}K"
    return f"{value:.4g}"


def main() -> None:
    args = parse_args()
    series = [read_series(path, args.x, args.metric) for path in args.log]
    series = [(name, points) for name, points in series if points]
    if not series:
        raise ValueError(f"no finite {args.metric!r} points were found")

    all_points = [point for _, points in series for point in points]
    x_min = min(point[0] for point in all_points)
    x_max = max(point[0] for point in all_points)
    y_min = min(point[1] for point in all_points)
    y_max = max(point[1] for point in all_points)
    if x_min == x_max:
        x_max = x_min + 1
    if y_min == y_max:
        y_min -= 0.5
        y_max += 0.5
    y_padding = 0.05 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    width, height = args.width, args.height
    left, right, top, bottom = 90, 260, 70, 80
    plot_width = width - left - right
    plot_height = height - top - bottom

    def sx(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def sy(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="34" text-anchor="middle" font-family="sans-serif" font-size="22">{escape(args.title)}</text>',
    ]
    for value in ticks(x_min, x_max):
        x = sx(value)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 28}" text-anchor="middle" font-family="sans-serif" font-size="12">{escape(format_number(value))}</text>'
        )
    for value in ticks(y_min, y_max):
        y = sy(value)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        elements.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{escape(format_number(value))}</text>'
        )
    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="15">{escape(args.x)}</text>',
            f'<text x="22" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 22 {top + plot_height / 2})" font-family="sans-serif" font-size="15">{escape(args.metric)}</text>',
        ]
    )
    for index, (name, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.2f},{sy(y):.2f}" for x, y in points)
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.5"/>')
        legend_y = top + 24 * index
        elements.append(
            f'<line x1="{width - right + 25}" y1="{legend_y}" x2="{width - right + 55}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{width - right + 65}" y="{legend_y + 4}" font-family="sans-serif" font-size="13">{escape(name)}</text>'
        )
    elements.append("</svg>")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(elements) + "\n", encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
