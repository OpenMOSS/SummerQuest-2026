#!/usr/bin/env python3
"""Render train/validation loss curves from one or more JSONL logs as SVG."""

from __future__ import annotations

import argparse
import html
import json
import math
import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath


COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
)
MAX_POINTS_PER_CURVE = 2_000
X_FIELDS = ("step", "wall_clock_sec")
X_AXIS_LABELS = {
    "step": "step",
    "wall_clock_sec": "wall clock (seconds)",
}
X_DESCRIPTIONS = {
    "step": "training step",
    "wall_clock_sec": "elapsed wall-clock time in seconds",
}
_UUID_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{8}(?:-[0-9a-f]{4}){3}-[0-9a-f]{12}(?![0-9a-f])")
_LONG_HEX_ID_RE = re.compile(r"(?i)(?<![0-9a-f])[0-9a-f]{20,}(?![0-9a-f])")
_UNIX_PATH_RE = re.compile(r"(?<![\w:/])/(?:[^/\s]+/)+[^\s,;:)\]}]*")
_WINDOWS_PATH_RE = re.compile(r"(?i)(?<![A-Za-z0-9_])[A-Z]:\\(?:[^\\\r\n]+\\)*[^\\\r\n\s,;:)\]}]*")


@dataclass(frozen=True)
class Series:
    label: str
    train: tuple[tuple[float, float], ...]
    validation: tuple[tuple[float, float], ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("jsonl", nargs="*", type=Path, help="Metrics JSONL files.")
    parser.add_argument(
        "--input",
        dest="extra_inputs",
        action="append",
        type=Path,
        default=[],
        help="Additional metrics JSONL file; may be repeated.",
    )
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output SVG path.")
    parser.add_argument("--title", default="Training and validation loss")
    parser.add_argument("--width", type=int, default=960)
    parser.add_argument("--height", type=int, default=560)
    parser.add_argument(
        "--x-field",
        choices=X_FIELDS,
        default="step",
        help="JSONL field to use for the x axis (default: step).",
    )
    return parser.parse_args()


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def read_series(path: Path, label: str, x_field: str = "step") -> Series:
    if x_field not in X_FIELDS:
        raise ValueError(f"unsupported x field: {x_field}")
    train: list[tuple[float, float]] = []
    validation: list[tuple[float, float]] = []
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as error:
        raise ValueError(f"could not read {path}: {error}") from error
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"invalid JSON in {path} at line {line_number}: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"expected an object in {path} at line {line_number}")
        x_value = finite_number(record.get(x_field))
        if x_value is None:
            continue
        train_loss = finite_number(record.get("train_loss"))
        val_loss = finite_number(record.get("val_loss"))
        if train_loss is not None:
            train.append((x_value, train_loss))
        if val_loss is not None:
            validation.append((x_value, val_loss))
    train.sort(key=lambda point: point[0])
    validation.sort(key=lambda point: point[0])
    return Series(label, tuple(train), tuple(validation))


def public_text(value: str) -> str:
    """Remove path and identifier-shaped metadata from public SVG labels."""

    windows_path = PureWindowsPath(value)
    if windows_path.is_absolute():
        return windows_path.name or "result"
    path = Path(value)
    if path.is_absolute():
        return path.name or "result"
    value = _UNIX_PATH_RE.sub("<redacted-path>", value)
    value = _WINDOWS_PATH_RE.sub("<redacted-path>", value)
    value = _UUID_RE.sub("redacted-id", value)
    return _LONG_HEX_ID_RE.sub("redacted-id", value)


def unique_labels(paths: list[Path]) -> list[str]:
    used: dict[str, int] = {}
    labels: list[str] = []
    for path in paths:
        base = path.parent.name if path.stem == "metrics" and path.parent.name else path.stem
        base = public_text(base)
        count = used.get(base, 0) + 1
        used[base] = count
        labels.append(base if count == 1 else f"{base} ({count})")
    return labels


def format_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 10_000 or (0 < magnitude < 0.001):
        return f"{value:.1e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 10:
        return f"{value:.1f}"
    return f"{value:.3g}"


def downsample(points: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if len(points) > MAX_POINTS_PER_CURVE:
        last = len(points) - 1
        indices = {round(index * last / (MAX_POINTS_PER_CURVE - 1)) for index in range(MAX_POINTS_PER_CURVE)}
        return tuple(points[index] for index in sorted(indices))
    return points


def polyline(
    points: tuple[tuple[float, float], ...],
    x_scale: Callable[[float], float],
    y_scale: Callable[[float], float],
) -> str:
    points = downsample(points)
    scaled = " ".join(f"{x_scale(x):.1f},{y_scale(y):.1f}" for x, y in points)
    return scaled


def render_svg(
    series: list[Series],
    width: int,
    height: int,
    title: str,
    x_field: str = "step",
) -> str:
    if x_field not in X_FIELDS:
        raise ValueError(f"unsupported x field: {x_field}")
    all_points = [point for item in series for point in item.train + item.validation]
    if not all_points:
        raise ValueError(f"no finite {x_field}/train_loss/val_loss points were found")

    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    if x_min == x_max:
        x_min -= 0.5
        x_max += 0.5
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.1)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding

    legend_items: list[tuple[str, str, bool]] = []
    for index, item in enumerate(series):
        color = COLORS[index % len(COLORS)]
        if item.train:
            legend_items.append((f"{item.label} train", color, False))
        if item.validation:
            legend_items.append((f"{item.label} val", color, True))

    left, right, bottom = 78.0, 28.0, 70.0
    legend_columns = max(1, min(4, int((width - left - right) // 190)))
    legend_rows = math.ceil(len(legend_items) / legend_columns)
    top = 58.0 + legend_rows * 20.0
    plot_width = width - left - right
    plot_height = height - top - bottom
    if plot_width < 300 or plot_height < 120:
        raise ValueError("canvas is too small for the number of curves; increase --width or --height")

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_scale(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    escaped_title = html.escape(title)
    style = (
        "text{font-family:ui-sans-serif,system-ui,sans-serif;fill:#334155}"
        ".grid{stroke:#e2e8f0;stroke-width:1}"
        ".axis{stroke:#64748b;stroke-width:1.2}"
        ".curve{fill:none;stroke-linejoin:round;stroke-linecap:round}"
        ".tick{font-size:12px}.legend{font-size:12px}.label{font-size:14px}"
        ".heading{font-size:20px;font-weight:600;fill:#0f172a}"
    )
    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        (
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
            f'viewBox="0 0 {width} {height}" role="img" aria-labelledby="title desc">'
        ),
        f'<title id="title">{escaped_title}</title>',
        (
            f'<desc id="desc">Loss by {X_DESCRIPTIONS[x_field]}, with solid training curves '
            "and dashed validation curves.</desc>"
        ),
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f"<style>{style}</style>",
        (
            f'<clipPath id="plot-area"><rect x="{left:.1f}" y="{top:.1f}" '
            f'width="{plot_width:.1f}" height="{plot_height:.1f}"/></clipPath>'
        ),
        f'<text class="heading" x="{left:.1f}" y="32">{escaped_title}</text>',
    ]

    legend_x = left
    legend_y = 58.0
    legend_column_width = plot_width / legend_columns
    for index, (label, color, dashed) in enumerate(legend_items):
        column = index % legend_columns
        row = index // legend_columns
        x = legend_x + column * legend_column_width
        y = legend_y + row * 20
        dash = ' stroke-dasharray="7 4"' if dashed else ""
        lines.append(
            f'<line x1="{x:.1f}" y1="{y - 4:.1f}" x2="{x + 25:.1f}" y2="{y - 4:.1f}" '
            f'stroke="{color}" stroke-width="2"{dash}/>'
        )
        lines.append(f'<text class="legend" x="{x + 31:.1f}" y="{y:.1f}">{html.escape(label)}</text>')

    ticks = 5
    for index in range(ticks + 1):
        fraction = index / ticks
        x = left + fraction * plot_width
        tick_value = x_min + fraction * (x_max - x_min)
        lines.append(f'<line class="grid" x1="{x:.1f}" y1="{top:.1f}" x2="{x:.1f}" y2="{top + plot_height:.1f}"/>')
        lines.append(
            f'<text class="tick" x="{x:.1f}" y="{top + plot_height + 23:.1f}" '
            f'text-anchor="middle">{html.escape(format_tick(tick_value))}</text>'
        )
    for index in range(ticks + 1):
        fraction = index / ticks
        y = top + fraction * plot_height
        loss = y_max - fraction * (y_max - y_min)
        lines.append(f'<line class="grid" x1="{left:.1f}" y1="{y:.1f}" x2="{left + plot_width:.1f}" y2="{y:.1f}"/>')
        lines.append(
            f'<text class="tick" x="{left - 11:.1f}" y="{y + 4:.1f}" text-anchor="end">{html.escape(format_tick(loss))}</text>'
        )
    lines.extend(
        (
            (
                f'<line class="axis" x1="{left:.1f}" y1="{top + plot_height:.1f}" '
                f'x2="{left + plot_width:.1f}" y2="{top + plot_height:.1f}"/>'
            ),
            (f'<line class="axis" x1="{left:.1f}" y1="{top:.1f}" x2="{left:.1f}" y2="{top + plot_height:.1f}"/>'),
            (
                f'<text class="label" x="{left + plot_width / 2:.1f}" y="{height - 20:.1f}" '
                f'text-anchor="middle">{X_AXIS_LABELS[x_field]}</text>'
            ),
            (
                f'<text class="label" x="20" y="{top + plot_height / 2:.1f}" text-anchor="middle" '
                f'transform="rotate(-90 20 {top + plot_height / 2:.1f})">loss</text>'
            ),
        )
    )

    lines.append('<g clip-path="url(#plot-area)">')
    for index, item in enumerate(series):
        color = COLORS[index % len(COLORS)]
        if item.train:
            lines.append(
                f'<polyline class="curve" points="{polyline(item.train, x_scale, y_scale)}" '
                f'stroke="{color}" stroke-width="1.7" opacity="0.78"/>'
            )
            if len(item.train) == 1:
                x_value, loss = item.train[0]
                lines.append(f'<circle cx="{x_scale(x_value):.1f}" cy="{y_scale(loss):.1f}" r="2.2" fill="{color}"/>')
        if item.validation:
            lines.append(
                f'<polyline class="curve" points="{polyline(item.validation, x_scale, y_scale)}" '
                f'stroke="{color}" stroke-width="2.4" stroke-dasharray="7 4"/>'
            )
            for x_value, loss in downsample(item.validation):
                lines.append(f'<circle cx="{x_scale(x_value):.1f}" cy="{y_scale(loss):.1f}" r="2.2" fill="{color}"/>')

    lines.append("</g>")
    lines.append("</svg>")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    paths = [*args.jsonl, *args.extra_inputs]
    if not paths:
        raise ValueError("provide at least one JSONL input")
    if args.width < 480 or args.height < 320:
        raise ValueError("--width must be at least 480 and --height at least 320")
    output_resolved = args.output.resolve()
    if output_resolved in {path.resolve() for path in paths}:
        raise ValueError("--output must not overwrite an input JSONL file")
    labels = unique_labels(paths)
    series = [read_series(path, label, args.x_field) for path, label in zip(paths, labels, strict=True)]
    svg = render_svg(series, args.width, args.height, public_text(args.title), args.x_field)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    temporary = args.output.with_name(f".{args.output.name}.tmp")
    temporary.write_text(svg, encoding="utf-8")
    temporary.replace(args.output)
    print(f"wrote {args.output} ({len(series)} run(s))")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
