from __future__ import annotations

import argparse
import json
import math
from html import escape
from pathlib import Path


COLORS = (
    "#2563eb",
    "#dc2626",
    "#16a34a",
    "#9333ea",
    "#ea580c",
    "#0891b2",
    "#4f46e5",
    "#be123c",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot train/validation loss curves from one or more metrics.jsonl files."
    )
    parser.add_argument("--metrics", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels in the same order as --metrics.",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--title", default="Language-model learning curves")
    return parser.parse_args()


def read_metrics(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    records: list[dict[str, object]] = []
    with path.open(encoding="utf-8") as metrics_file:
        for line_number, line in enumerate(metrics_file, 1):
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(
                    f"Invalid JSON in {path} on line {line_number}."
                ) from exc
            if isinstance(record, dict):
                records.append(record)
    return records


def finite_number(value: object) -> float | None:
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    return number if math.isfinite(number) else None


def collect_points(
    records: list[dict[str, object]],
    x_key: str,
    record_type: str,
) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    for record in records:
        if record.get("type") != record_type:
            continue
        x = finite_number(record.get(x_key))
        loss_key = "train_loss" if record_type == "train" else "val_loss"
        y = finite_number(record.get(loss_key, record.get("loss")))
        if x is not None and y is not None:
            points.append((x, y))
    return sorted(points)


def svg_polyline(
    points: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    plot_width: float,
    plot_height: float,
) -> str:
    def project(point: tuple[float, float]) -> tuple[float, float]:
        x, y = point
        svg_x = left + (x - x_min) / (x_max - x_min) * plot_width
        svg_y = top + (y_max - y) / (y_max - y_min) * plot_height
        return svg_x, svg_y

    return " ".join(f"{x:.2f},{y:.2f}" for x, y in map(project, points))


def write_loss_svg(
    output_path: Path,
    series: list[tuple[str, list[dict[str, object]]]],
    x_key: str,
    x_label: str,
    title: str,
) -> None:
    width = 1000
    height = 650
    left = 90.0
    right = 260.0
    top = 70.0
    bottom = 80.0
    plot_width = width - left - right
    plot_height = height - top - bottom

    plotted: list[tuple[str, str, list[tuple[float, float]], str]] = []
    all_points: list[tuple[float, float]] = []
    for index, (label, records) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        for record_type, dash in (("train", ""), ("validation", "8 5")):
            points = collect_points(records, x_key=x_key, record_type=record_type)
            if points:
                plotted.append((label, record_type, points, color))
                all_points.extend(points)

    if not all_points:
        raise ValueError(f"No loss points with x key {x_key!r} were found.")

    x_values = [point[0] for point in all_points]
    y_values = [point[1] for point in all_points]
    x_min = min(x_values)
    x_max = max(x_values)
    y_min = min(y_values)
    y_max = max(y_values)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    y_padding = 0.05 * (y_max - y_min)
    y_min -= y_padding
    y_max += y_padding

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="32" text-anchor="middle" font-family="sans-serif" font-size="22">{escape(title)}</text>',
    ]

    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * plot_width
        y = top + fraction * plot_height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        parts.extend(
            [
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12">{x_value:.4g}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{y_value:.4g}</text>',
            ]
        )

    parts.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 22}" text-anchor="middle" font-family="sans-serif" font-size="15">{escape(x_label)}</text>',
            f'<text x="24" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 24 {top + plot_height / 2})" font-family="sans-serif" font-size="15">Cross-entropy loss</text>',
        ]
    )

    legend_y = top
    for label, record_type, points, color in plotted:
        coordinates = svg_polyline(
            points,
            x_min,
            x_max,
            y_min,
            y_max,
            left,
            top,
            plot_width,
            plot_height,
        )
        dash = ' stroke-dasharray="8 5"' if record_type == "validation" else ""
        parts.append(
            f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.2"{dash}/>'
        )
        if record_type == "validation":
            for x, y in points:
                projected = svg_polyline(
                    [(x, y)],
                    x_min,
                    x_max,
                    y_min,
                    y_max,
                    left,
                    top,
                    plot_width,
                    plot_height,
                )
                point_x, point_y = projected.split(",")
                parts.append(
                    f'<circle cx="{point_x}" cy="{point_y}" r="3.2" fill="{color}"/>'
                )

        display_label = f"{label} — {record_type}"
        parts.extend(
            [
                f'<line x1="{left + plot_width + 28}" y1="{legend_y}" x2="{left + plot_width + 64}" y2="{legend_y}" stroke="{color}" stroke-width="2.2"{dash}/>',
                f'<text x="{left + plot_width + 72}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(display_label)}</text>',
            ]
        )
        legend_y += 24

    parts.append("</svg>")
    output_path.write_text("\n".join(parts), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.labels is not None and len(args.labels) != len(args.metrics):
        raise ValueError("--labels must contain one label per metrics file.")

    labels = args.labels or [path.parent.name for path in args.metrics]
    series = [
        (label, read_metrics(path))
        for label, path in zip(labels, args.metrics, strict=True)
    ]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    outputs = (
        (
            args.output_dir / "loss_vs_step.svg",
            "step",
            "Optimizer step",
        ),
        (
            args.output_dir / "loss_vs_wall_time.svg",
            "wall_clock_sec",
            "Cumulative wall-clock time (seconds)",
        ),
        (
            args.output_dir / "loss_vs_tokens.svg",
            "tokens_processed",
            "Training tokens processed",
        ),
    )
    for output_path, x_key, x_label in outputs:
        if output_path.exists():
            raise FileExistsError(f"Refusing to overwrite: {output_path}")
        write_loss_svg(
            output_path=output_path,
            series=series,
            x_key=x_key,
            x_label=x_label,
            title=args.title,
        )
        print(f"curve saved: {output_path}")


if __name__ == "__main__":
    main()
