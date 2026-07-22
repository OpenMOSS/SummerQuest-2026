from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from html import escape
from pathlib import Path


@dataclass(frozen=True)
class CurveSpec:
    label: str
    path: str
    color: str


LR_CURVES = (
    CurveSpec("LR 3e-4", "logs/train_tinystories.jsonl", "#7c3aed"),
    CurveSpec("LR 6e-4", "logs/lr_sweep/train_tinystories_lr_6e-4.jsonl", "#2563eb"),
    CurveSpec("LR 1.2e-3", "logs/lr_sweep/train_tinystories_lr_1p2e-3.jsonl", "#16a34a"),
    CurveSpec("LR 4.8e-3 (diverged)", "logs/lr_sweep/train_tinystories_lr_4p8e-3.jsonl", "#dc2626"),
)

BATCH_CURVES = (
    CurveSpec("Batch 128", "logs/lr_sweep/train_tinystories_lr_1p2e-3.jsonl", "#111827"),
    CurveSpec("Batch 256", "logs/batch_size_power2_clean/train_tinystories_bs_256.jsonl", "#9333ea"),
    CurveSpec("Batch 64", "logs/batch_size/train_tinystories_bs_64.jsonl", "#16a34a"),
    CurveSpec("Batch 32", "logs/batch_size/train_tinystories_bs_32.jsonl", "#2563eb"),
    CurveSpec("Batch 1 (partial)", "logs/batch_size/train_tinystories_bs_1.jsonl", "#d97706"),
)

ABLATION_CURVES = (
    CurveSpec("Baseline", "logs/lr_sweep/train_tinystories_lr_1p2e-3.jsonl", "#111827"),
    CurveSpec("Post-Norm", "logs/ablation/train_tinystories_ablation_post_norm.jsonl", "#2563eb"),
    CurveSpec("SiLU FFN", "logs/ablation/train_tinystories_ablation_silu.jsonl", "#16a34a"),
    CurveSpec("NoPE (diverged)", "logs/ablation/train_tinystories_ablation_nope.jsonl", "#dc2626"),
    CurveSpec("No RMSNorm (diverged)", "logs/ablation/train_tinystories_ablation_no_rmsnorm.jsonl", "#9333ea"),
)

OWT_CURVES = (CurveSpec("OWT", "logs/owt/train_owt.jsonl", "#2563eb"),)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render training JSONL logs as dependency-free SVG plots.")
    parser.add_argument("--output-dir", default="plots")
    parser.add_argument("--smoothing-window", type=int, default=25)
    return parser


def _read_records(path: Path) -> list[dict[str, object]]:
    records: list[dict[str, object]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                value = json.loads(line)
                if isinstance(value, dict):
                    records.append(value)
    return records


def _finite_number(value: object) -> float | None:
    if not isinstance(value, (int, float)):
        return None
    result = float(value)
    return result if math.isfinite(result) else None


def _moving_average(points: list[tuple[float, float]], window: int) -> list[tuple[float, float]]:
    if window <= 1:
        return points
    smoothed: list[tuple[float, float]] = []
    running_sum = 0.0
    values: list[float] = []
    for x, y in points:
        values.append(y)
        running_sum += y
        if len(values) > window:
            running_sum -= values[-window - 1]
        smoothed.append((x, running_sum / min(len(values), window)))
    return smoothed


def _extract_curves(
    spec: CurveSpec,
    smoothing_window: int,
) -> tuple[list[tuple[float, float]], list[tuple[float, float]]]:
    train_points: list[tuple[float, float]] = []
    validation_points: list[tuple[float, float]] = []
    for record in _read_records(Path(spec.path)):
        processed_tokens = _finite_number(record.get("processed_tokens"))
        if processed_tokens is None:
            continue
        x = processed_tokens / 1_000_000
        train_loss = _finite_number(record.get("train_loss"))
        if train_loss is not None:
            train_points.append((x, train_loss))
        validation_loss = _finite_number(record.get("val_loss"))
        if validation_loss is not None:
            validation_points.append((x, validation_loss))
    return _moving_average(train_points, smoothing_window), validation_points


def _tick_values(low: float, high: float, count: int = 5) -> list[float]:
    return [low + index * (high - low) / count for index in range(count + 1)]


def _polyline(
    points: list[tuple[float, float]],
    *,
    x_scale: callable,
    y_scale: callable,
) -> str:
    return " ".join(f"{x_scale(x):.2f},{y_scale(y):.2f}" for x, y in points)


def _render_plot(
    *,
    title: str,
    subtitle: str,
    specs: tuple[CurveSpec, ...],
    output_path: Path,
    smoothing_window: int,
    y_min: float,
    y_max: float,
) -> None:
    width = 1000
    height = 600
    left = 82
    right = 230
    top = 78
    bottom = 72
    plot_width = width - left - right
    plot_height = height - top - bottom

    curves = [(spec, *_extract_curves(spec, smoothing_window)) for spec in specs if Path(spec.path).exists()]
    all_points = [point for _, train, validation in curves for point in (*train, *validation)]
    if not all_points:
        raise ValueError(f"No plot data for {title}.")
    x_min = 0.0
    x_max = max(point[0] for point in all_points)
    if x_max <= x_min:
        x_max = x_min + 1.0

    def x_scale(value: float) -> float:
        return left + (value - x_min) / (x_max - x_min) * plot_width

    def y_scale(value: float) -> float:
        clipped = min(max(value, y_min), y_max)
        return top + (y_max - clipped) / (y_max - y_min) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="700" fill="#111827">{escape(title)}</text>',
        f'<text x="{left}" y="57" font-family="sans-serif" font-size="13" fill="#4b5563">{escape(subtitle)}</text>',
    ]

    for tick in _tick_values(y_min, y_max):
        y = y_scale(tick)
        elements.append(f'<line x1="{left}" y1="{y:.2f}" x2="{left + plot_width}" y2="{y:.2f}" stroke="#e5e7eb"/>')
        elements.append(
            f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" font-family="sans-serif" font-size="12" fill="#4b5563">{tick:.2f}</text>'
        )
    for tick in _tick_values(x_min, x_max):
        x = x_scale(tick)
        elements.append(f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + plot_height}" stroke="#f3f4f6"/>')
        elements.append(
            f'<text x="{x:.2f}" y="{top + plot_height + 25}" text-anchor="middle" font-family="sans-serif" font-size="12" fill="#4b5563">{tick:.0f}</text>'
        )

    elements.extend(
        [
            f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
            f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">Processed tokens (millions)</text>',
            f'<text x="22" y="{top + plot_height / 2}" transform="rotate(-90 22 {top + plot_height / 2})" text-anchor="middle" font-family="sans-serif" font-size="13" fill="#111827">Cross-entropy loss</text>',
        ]
    )

    for spec, train_points, validation_points in curves:
        if train_points:
            elements.append(
                f'<polyline points="{_polyline(train_points, x_scale=x_scale, y_scale=y_scale)}" fill="none" stroke="{spec.color}" stroke-width="1.5" opacity="0.35"/>'
            )
        if validation_points:
            elements.append(
                f'<polyline points="{_polyline(validation_points, x_scale=x_scale, y_scale=y_scale)}" fill="none" stroke="{spec.color}" stroke-width="2.5"/>'
            )
            for x_value, y_value in validation_points:
                elements.append(
                    f'<circle cx="{x_scale(x_value):.2f}" cy="{y_scale(y_value):.2f}" r="2.4" fill="{spec.color}"/>'
                )

    legend_x = left + plot_width + 24
    legend_y = top + 8
    for index, (spec, _, _) in enumerate(curves):
        y = legend_y + index * 30
        elements.append(
            f'<line x1="{legend_x}" y1="{y}" x2="{legend_x + 25}" y2="{y}" stroke="{spec.color}" stroke-width="3"/>'
        )
        elements.append(
            f'<text x="{legend_x + 34}" y="{y + 4}" font-family="sans-serif" font-size="12" fill="#111827">{escape(spec.label)}</text>'
        )
    elements.append(
        f'<text x="{legend_x}" y="{legend_y + len(curves) * 30 + 22}" font-family="sans-serif" font-size="11" fill="#6b7280">faint: smoothed train</text>'
    )
    elements.append(
        f'<text x="{legend_x}" y="{legend_y + len(curves) * 30 + 38}" font-family="sans-serif" font-size="11" fill="#6b7280">solid + dots: validation</text>'
    )
    elements.append("</svg>")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(elements) + "\n", encoding="utf-8")


def main() -> None:
    args = _build_parser().parse_args()
    if args.smoothing_window <= 0:
        raise ValueError("smoothing_window must be positive.")
    output_dir = Path(args.output_dir)
    _render_plot(
        title="TinyStories learning-rate sweep",
        subtitle="Same architecture, batch 128, and 327.68M-token budget",
        specs=LR_CURVES,
        output_path=output_dir / "lr_sweep.svg",
        smoothing_window=args.smoothing_window,
        y_min=1.2,
        y_max=5.0,
    )
    _render_plot(
        title="TinyStories batch-size comparison",
        subtitle="Fixed 327.68M-token budget where completed; batch 1 partial, batch 512 OOM",
        specs=BATCH_CURVES,
        output_path=output_dir / "batch_size.svg",
        smoothing_window=args.smoothing_window,
        y_min=1.2,
        y_max=5.0,
    )
    _render_plot(
        title="TinyStories architecture ablations",
        subtitle="Same optimizer settings and token budget; divergent runs stop early",
        specs=ABLATION_CURVES,
        output_path=output_dir / "ablations.svg",
        smoothing_window=args.smoothing_window,
        y_min=1.2,
        y_max=10.0,
    )
    _render_plot(
        title="OpenWebText training",
        subtitle="4-layer model, batch 128, 10,000 steps, 327.68M tokens",
        specs=OWT_CURVES,
        output_path=output_dir / "owt_training.svg",
        smoothing_window=args.smoothing_window,
        y_min=3.5,
        y_max=10.5,
    )
    print(f"Wrote plots to {output_dir}")


if __name__ == "__main__":
    main()
