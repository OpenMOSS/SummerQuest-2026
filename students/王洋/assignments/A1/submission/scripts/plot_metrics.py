#!/usr/bin/env python3
"""Render one or more training JSONL logs as a dependency-free SVG plot."""

from __future__ import annotations

import argparse
import html
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path


COLORS = ("#2563eb", "#dc2626", "#059669", "#9333ea", "#ea580c", "#0891b2", "#4f46e5", "#65a30d")


@dataclass(frozen=True)
class Point:
    x: float
    loss: float
    step: int
    processed_tokens: int
    wall_time_seconds: float


@dataclass(frozen=True)
class Run:
    label: str
    train: list[Point]
    validation: list[Point]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot train/validation loss from one or more metrics.jsonl files.")
    parser.add_argument("metrics", type=Path, nargs="+", help="one or more metrics.jsonl files")
    parser.add_argument("--output", type=Path, required=True, help="destination .svg file")
    parser.add_argument(
        "--x-axis",
        choices=("step", "processed_tokens", "wall_time_seconds"),
        default="step",
    )
    parser.add_argument("--label", action="append", help="run label; repeat once per metrics file")
    parser.add_argument("--title", default="Language-model loss")
    parser.add_argument("--width", type=int, default=1000)
    parser.add_argument("--height", type=int, default=640)
    parser.add_argument("--max-points", type=int, default=2000, help="maximum plotted points per series")
    parser.add_argument("--y-min", type=float, help="optional fixed lower loss bound")
    parser.add_argument("--y-max", type=float, help="optional fixed upper loss bound")
    return parser.parse_args()


def load_run(path: Path, label: str, x_axis: str) -> Run:
    series: dict[str, list[Point]] = {"train": [], "validation": []}
    with path.open(encoding="utf-8") as input_file:
        for line_number, line in enumerate(input_file, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {error}") from error
            event_type = event.get("event")
            if event_type not in series or "loss" not in event:
                continue
            try:
                loss = float(event["loss"])
                x = float(event[x_axis])
                step = int(event["step"])
                processed_tokens = int(event["processed_tokens"])
                wall_time_seconds = float(event["wall_time_seconds"])
            except (KeyError, TypeError, ValueError) as error:
                raise ValueError(f"malformed loss event at {path}:{line_number}") from error
            if math.isfinite(loss) and math.isfinite(x):
                series[event_type].append(Point(x, loss, step, processed_tokens, wall_time_seconds))

    for points in series.values():
        points.sort(key=lambda point: point.x)
    if not series["train"] and not series["validation"]:
        raise ValueError(f"no finite train or validation loss events found in {path}")
    return Run(label=label, train=series["train"], validation=series["validation"])


def downsample(points: list[Point], maximum: int) -> list[Point]:
    if len(points) <= maximum:
        return points
    if maximum < 2:
        return [points[-1]]
    indices = [round(index * (len(points) - 1) / (maximum - 1)) for index in range(maximum)]
    return [points[index] for index in indices]


def format_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000_000:
        return f"{value / 1_000_000_000:.2g}B"
    if magnitude >= 1_000_000:
        return f"{value / 1_000_000:.2g}M"
    if magnitude >= 1_000:
        return f"{value / 1_000:.2g}K"
    if magnitude >= 10:
        return f"{value:.0f}"
    return f"{value:.3g}"


def svg_text(x: float, y: float, value: str, **attributes: str) -> str:
    rendered_attributes = " ".join(f'{key.replace("_", "-")}="{html.escape(item)}"' for key, item in attributes.items())
    return f'<text x="{x:.2f}" y="{y:.2f}" {rendered_attributes}>{html.escape(value)}</text>'


def render_svg(
    runs: list[Run],
    *,
    width: int,
    height: int,
    title: str,
    x_axis: str,
    max_points: int,
    fixed_y_min: float | None,
    fixed_y_max: float | None,
) -> str:
    left, right, top, bottom = 90.0, 30.0, 65.0, 85.0
    legend_height = 28.0 * len(runs)
    bottom = max(bottom, legend_height + 90.0)
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("width and height are too small for the plot margins")

    all_points = [point for run in runs for points in (run.train, run.validation) for point in points]
    x_min = min(point.x for point in all_points)
    x_max = max(point.x for point in all_points)
    y_min = min(point.loss for point in all_points) if fixed_y_min is None else fixed_y_min
    y_max = max(point.loss for point in all_points) if fixed_y_max is None else fixed_y_max
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min >= y_max:
        if y_min == y_max and fixed_y_min is None:
            y_min -= 0.5
        elif y_min == y_max and fixed_y_max is None:
            y_max += 0.5
        else:
            raise ValueError("y-min must be smaller than y-max")
    if fixed_y_min is None or fixed_y_max is None:
        padding = 0.05 * (y_max - y_min)
        if fixed_y_min is None:
            y_min -= padding
        if fixed_y_max is None:
            y_max += padding

    def project_x(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def project_y(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    parts = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text { font-family: ui-sans-serif, system-ui, sans-serif; fill: #172033; }",
        ".grid { stroke: #d8dee9; stroke-width: 1; }",
        ".axis { stroke: #475569; stroke-width: 1.4; }",
        ".series { fill: none; stroke-width: 2; stroke-linejoin: round; stroke-linecap: round; }",
        "</style>",
        f'<defs><clipPath id="plot-area"><rect x="{left}" y="{top}" '
        f'width="{plot_width}" height="{plot_height}"/></clipPath></defs>',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        svg_text(width / 2, 32, title, text_anchor="middle", font_size="20", font_weight="600"),
    ]

    tick_count = 6
    for index in range(tick_count):
        fraction = index / (tick_count - 1)
        x_value = x_min + fraction * (x_max - x_min)
        x = project_x(x_value)
        parts.append(f'<line class="grid" x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}"/>')
        parts.append(svg_text(x, top + plot_height + 24, format_tick(x_value), text_anchor="middle", font_size="12"))

        y_value = y_min + fraction * (y_max - y_min)
        y = project_y(y_value)
        parts.append(f'<line class="grid" x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}"/>')
        parts.append(svg_text(left - 12, y + 4, format_tick(y_value), text_anchor="end", font_size="12"))

    parts.extend(
        [
            f'<line class="axis" x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}"/>',
            f'<line class="axis" x1="{left}" y1="{top + plot_height}" '
            f'x2="{left + plot_width}" y2="{top + plot_height}"/>',
            svg_text(
                left + plot_width / 2,
                top + plot_height + 51,
                x_axis.replace("_", " "),
                text_anchor="middle",
                font_size="14",
            ),
            f'<text x="20" y="{top + plot_height / 2:.2f}" text-anchor="middle" font-size="14" '
            f'transform="rotate(-90 20 {top + plot_height / 2:.2f})">loss</text>',
        ]
    )

    for run_index, run in enumerate(runs):
        color = COLORS[run_index % len(COLORS)]
        for event_type, points in (("train", run.train), ("validation", run.validation)):
            sampled = downsample(points, max_points)
            if not sampled:
                continue
            coordinates = " ".join(f"{project_x(point.x):.2f},{project_y(point.loss):.2f}" for point in sampled)
            dash = ' stroke-dasharray="7 5"' if event_type == "validation" else ""
            parts.append(
                f'<polyline class="series" stroke="{color}"{dash} points="{coordinates}" clip-path="url(#plot-area)"/>'
            )
            if event_type == "validation":
                for point in sampled:
                    parts.append(
                        f'<circle cx="{project_x(point.x):.2f}" cy="{project_y(point.loss):.2f}" '
                        f'r="2.8" fill="{color}" clip-path="url(#plot-area)"><title>'
                        f"{html.escape(run.label)} validation: "
                        f"{point.loss:.6g}</title></circle>"
                    )

        legend_y = height - bottom + 75 + 28 * run_index
        parts.append(
            f'<line x1="{left}" y1="{legend_y}" x2="{left + 27}" y2="{legend_y}" stroke="{color}" stroke-width="2"/>'
        )
        parts.append(
            f'<line x1="{left + 37}" y1="{legend_y}" x2="{left + 64}" y2="{legend_y}" '
            f'stroke="{color}" stroke-width="2" stroke-dasharray="7 5"/>'
        )
        parts.append(
            svg_text(
                left + 75,
                legend_y + 4,
                f"{run.label} (solid: train, dashed: validation)",
                font_size="12",
            )
        )

    parts.append("</svg>")
    return "\n".join(parts) + "\n"


def main() -> None:
    args = parse_args()
    if args.output.suffix.lower() != ".svg":
        raise ValueError("--output must end in .svg")
    if args.width < 400 or args.height < 300:
        raise ValueError("width must be at least 400 and height at least 300")
    if args.max_points <= 0:
        raise ValueError("max-points must be positive")
    if args.label is not None and len(args.label) != len(args.metrics):
        raise ValueError("repeat --label exactly once for each metrics file")

    labels = args.label or [path.parent.name or path.stem for path in args.metrics]
    runs = [load_run(path, label, args.x_axis) for path, label in zip(args.metrics, labels, strict=True)]
    svg = render_svg(
        runs,
        width=args.width,
        height=args.height,
        title=args.title,
        x_axis=args.x_axis,
        max_points=args.max_points,
        fixed_y_min=args.y_min,
        fixed_y_max=args.y_max,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as output_file:
            output_file.write(svg)
            output_file.flush()
            os.fsync(output_file.fileno())
        os.replace(temporary, args.output)
    finally:
        temporary.unlink(missing_ok=True)
    print(args.output)


if __name__ == "__main__":
    main()
