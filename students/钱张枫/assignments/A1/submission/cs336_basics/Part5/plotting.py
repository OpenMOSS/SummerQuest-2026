from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from html import escape
from pathlib import Path
from typing import Any


@dataclass(frozen=True, slots=True)
class CurveSeries:
    label: str
    metrics_path: Path
    metric_key: str


def load_curve_points(series: CurveSeries, *, x_key: str) -> list[tuple[float, float]]:
    points: list[tuple[float, float]] = []
    with series.metrics_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            value: Any = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{series.metrics_path}:{line_number} must contain a JSON object.")
            x_value = value.get(x_key)
            y_value = value.get(series.metric_key)
            if y_value is None:
                continue
            if isinstance(x_value, bool) or not isinstance(x_value, (int, float)):
                raise ValueError(f"{series.metrics_path}:{line_number} has invalid {x_key}.")
            if isinstance(y_value, bool) or not isinstance(y_value, (int, float)):
                raise ValueError(f"{series.metrics_path}:{line_number} has invalid {series.metric_key}.")
            points.append((float(x_value), float(y_value)))
    if not points:
        raise ValueError(f"no {series.metric_key} points found in {series.metrics_path}.")
    return points


def render_loss_svg(
    series_list: list[CurveSeries],
    output_path: str | Path,
    *,
    x_key: str = "step",
    title: str = "Loss curve",
    width: int = 1000,
    height: int = 620,
) -> Path:
    if not series_list:
        raise ValueError("at least one curve series is required.")
    if width < 400 or height < 300:
        raise ValueError("SVG dimensions are too small for a readable chart.")

    series_points = [(series, load_curve_points(series, x_key=x_key)) for series in series_list]
    all_points = [point for _, points in series_points for point in points]
    min_x, max_x = _expanded_range(point[0] for point in all_points)
    min_y, max_y = _expanded_range(point[1] for point in all_points)

    margin_left = 90
    margin_right = 250
    margin_top = 70
    margin_bottom = 80
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    if plot_width <= 0 or plot_height <= 0:
        raise ValueError("SVG dimensions leave no room for the plot area.")

    def scale_x(value: float) -> float:
        return margin_left + (value - min_x) / (max_x - min_x) * plot_width

    def scale_y(value: float) -> float:
        return margin_top + (max_y - value) / (max_y - min_y) * plot_height

    colors = ["#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#ff7f0e", "#17becf", "#8c564b"]
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2:.1f}" y="36" text-anchor="middle" font-family="sans-serif" font-size="22">{escape(title)}</text>',
    ]

    for tick_index in range(6):
        fraction = tick_index / 5
        x_value = min_x + fraction * (max_x - min_x)
        x_position = scale_x(x_value)
        elements.append(
            f'<line x1="{x_position:.2f}" y1="{margin_top}" x2="{x_position:.2f}" y2="{margin_top + plot_height}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{x_position:.2f}" y="{margin_top + plot_height + 28}" text-anchor="middle" font-family="sans-serif" font-size="12">{_format_tick(x_value)}</text>'
        )

        y_value = min_y + fraction * (max_y - min_y)
        y_position = scale_y(y_value)
        elements.append(
            f'<line x1="{margin_left}" y1="{y_position:.2f}" x2="{margin_left + plot_width}" y2="{y_position:.2f}" stroke="#e5e7eb"/>'
        )
        elements.append(
            f'<text x="{margin_left - 12}" y="{y_position + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12">{_format_tick(y_value)}</text>'
        )

    elements.extend(
        [
            f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#111827" stroke-width="1.5"/>',
            f'<text x="{margin_left + plot_width / 2:.2f}" y="{height - 24}" text-anchor="middle" font-family="sans-serif" font-size="14">{escape(x_key)}</text>',
            f'<text x="24" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})" font-family="sans-serif" font-size="14">loss</text>',
        ]
    )

    legend_x = margin_left + plot_width + 28
    for index, (series, points) in enumerate(series_points):
        color = colors[index % len(colors)]
        polyline_points = " ".join(f"{scale_x(x):.2f},{scale_y(y):.2f}" for x, y in points)
        elements.append(
            f'<polyline points="{polyline_points}" fill="none" stroke="{color}" stroke-width="2" stroke-linejoin="round" stroke-linecap="round"/>'
        )
        legend_y = margin_top + 24 * index
        elements.append(
            f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{legend_x + 32}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">{escape(series.label)}</text>'
        )

    elements.append("</svg>")
    destination = Path(output_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text("\n".join(elements) + "\n", encoding="utf-8")
    return destination


def _expanded_range(values: Iterable[float]) -> tuple[float, float]:
    normalized_values = list(values)
    minimum = min(normalized_values)
    maximum = max(normalized_values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
        return minimum - padding, maximum + padding
    padding = (maximum - minimum) * 0.05
    return minimum - padding, maximum + padding


def _format_tick(value: float) -> str:
    magnitude = abs(value)
    if magnitude >= 1_000_000 or (0 < magnitude < 0.001):
        return f"{value:.2e}"
    if magnitude >= 100:
        return f"{value:.0f}"
    if magnitude >= 1:
        return f"{value:.2f}"
    return f"{value:.4f}"
