from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


COLORS = (
    "#2563eb",
    "#dc2626",
    "#059669",
    "#7c3aed",
    "#ea580c",
    "#0891b2",
    "#be185d",
    "#4d7c0f",
)


@dataclass(frozen=True)
class RunSeries:
    label: str
    train: list[tuple[float, float]]
    validation: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare train/validation loss JSONL logs in a standalone SVG.")
    parser.add_argument(
        "--input",
        type=Path,
        nargs="+",
        required=True,
        help="One or more train.jsonl files (a run directory is also accepted).",
    )
    parser.add_argument(
        "--label",
        action="append",
        default=[],
        help="Run label, repeated once per input. Defaults to the run directory name.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Training loss comparison")
    parser.add_argument("--x-axis", choices=("step", "processed_tokens"), default="step")
    return parser.parse_args()


def _resolve_log_path(path: Path) -> Path:
    return path / "train.jsonl" if path.is_dir() else path


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            value = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number}: invalid JSON: {error.msg}") from error
        if not isinstance(value, dict):
            raise ValueError(f"{path}:{line_number}: each JSONL record must be an object")
        records.append(value)
    return records


def _tokens_per_step(log_path: Path) -> int | None:
    summary_path = log_path.with_name("summary.json")
    if not summary_path.exists():
        return None
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    batch_size = summary.get("batch_size")
    context_length = summary.get("context_length")
    if batch_size is None or context_length is None:
        return None
    return int(batch_size) * int(context_length)


def _x_value(record: dict[str, Any], x_axis: str, tokens_per_step: int | None, path: Path) -> float:
    if x_axis == "step":
        if "step" not in record:
            raise ValueError(f"{path}: a loss record is missing step")
        return float(record["step"])
    if "processed_tokens" in record:
        return float(record["processed_tokens"])
    if "step" in record and tokens_per_step is not None:
        return float(record["step"]) * tokens_per_step
    raise ValueError(
        f"{path}: processed_tokens is missing and cannot be inferred; "
        "provide batch_size and context_length in the adjacent summary.json"
    )


def load_run(path: Path, label: str, x_axis: str) -> RunSeries:
    log_path = _resolve_log_path(path)
    records = _read_jsonl(log_path)
    tokens_per_step = _tokens_per_step(log_path) if x_axis == "processed_tokens" else None
    train = [
        (_x_value(record, x_axis, tokens_per_step, log_path), float(record["train_loss"]))
        for record in records
        if "train_loss" in record
    ]
    validation = [
        (_x_value(record, x_axis, tokens_per_step, log_path), float(record["val_loss"]))
        for record in records
        if "val_loss" in record
    ]
    if not train and not validation:
        raise ValueError(f"{log_path}: no train_loss or val_loss records")
    return RunSeries(label=label, train=train, validation=validation)


def _points(
    values: list[tuple[float, float]],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    *,
    width: float,
    height: float,
    left: float,
    top: float,
) -> str:
    return " ".join(
        f"{left + (x - x_min) / max(x_max - x_min, 1e-12) * width:.2f},"
        f"{top + (y_max - y) / max(y_max - y_min, 1e-12) * height:.2f}"
        for x, y in values
    )


def _format_tick(value: float) -> str:
    absolute = abs(value)
    if absolute >= 1_000_000_000:
        return f"{value / 1_000_000_000:.3g}B"
    if absolute >= 1_000_000:
        return f"{value / 1_000_000:.3g}M"
    if absolute >= 1_000:
        return f"{value / 1_000:.3g}k"
    return f"{value:g}"


def render_svg(runs: list[RunSeries], title: str, x_axis: str) -> str:
    all_values = [point for run in runs for points in (run.train, run.validation) for point in points]
    x_min, x_max = min(x for x, _ in all_values), max(x for x, _ in all_values)
    y_min, y_max = min(y for _, y in all_values), max(y for _, y in all_values)
    if y_min == y_max:
        padding = max(abs(y_min) * 0.05, 0.1)
        y_min -= padding
        y_max += padding
    else:
        padding = (y_max - y_min) * 0.05
        y_min -= padding
        y_max += padding

    canvas_width = 1100
    legend_rows = (len(runs) + 3) // 4
    canvas_height = 610 + max(0, legend_rows - 1) * 24
    left, top, width, height = 95.0, 75.0, 910.0, 420.0
    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{canvas_width}" height="{canvas_height}" '
        f'viewBox="0 0 {canvas_width} {canvas_height}">',
        f'<rect width="{canvas_width}" height="{canvas_height}" fill="white"/>',
        f'<text x="550" y="35" text-anchor="middle" font-family="sans-serif" font-size="24">'
        f"{html.escape(title)}</text>",
    ]

    for tick in range(6):
        fraction = tick / 5
        x = left + fraction * width
        y = top + fraction * height
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        elements.extend(
            (
                f'<line x1="{x:.2f}" y1="{top}" x2="{x:.2f}" y2="{top + height}" stroke="#e5e7eb"/>',
                f'<text x="{x:.2f}" y="{top + height + 23}" text-anchor="middle" '
                f'font-family="sans-serif" font-size="12">{html.escape(_format_tick(x_value))}</text>',
                f'<line x1="{left}" y1="{y:.2f}" x2="{left + width}" y2="{y:.2f}" stroke="#e5e7eb"/>',
                f'<text x="{left - 12}" y="{y + 4:.2f}" text-anchor="end" '
                f'font-family="sans-serif" font-size="12">{y_value:.3f}</text>',
            )
        )

    elements.extend(
        (
            f'<line x1="{left}" y1="{top + height}" x2="{left + width}" y2="{top + height}" stroke="black"/>',
            f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + height}" stroke="black"/>',
            f'<text x="{left + width / 2}" y="{top + height + 60}" text-anchor="middle" '
            f'font-family="sans-serif">{"processed tokens" if x_axis == "processed_tokens" else "step"}</text>',
            f'<text x="28" y="{top + height / 2}" text-anchor="middle" font-family="sans-serif" '
            f'transform="rotate(-90 28 {top + height / 2})">loss</text>',
        )
    )

    for index, run in enumerate(runs):
        color = COLORS[index % len(COLORS)]
        for values, dash, metric in (
            (run.train, "", "train"),
            (run.validation, ' stroke-dasharray="7 5"', "validation"),
        ):
            if not values:
                continue
            point_text = _points(
                values,
                x_min,
                x_max,
                y_min,
                y_max,
                width=width,
                height=height,
                left=left,
                top=top,
            )
            elements.append(
                f'<polyline fill="none" stroke="{color}" stroke-width="2"{dash} points="{point_text}">'
                f"<title>{html.escape(run.label)} {metric}</title></polyline>"
            )

        column = index % 4
        row = index // 4
        legend_x = 95 + column * 245
        legend_y = 585 + row * 24
        elements.extend(
            (
                f'<line x1="{legend_x}" y1="{legend_y - 5}" x2="{legend_x + 28}" y2="{legend_y - 5}" '
                f'stroke="{color}" stroke-width="3"/>',
                f'<text x="{legend_x + 36}" y="{legend_y}" font-family="sans-serif" font-size="13">'
                f"{html.escape(run.label)}</text>",
            )
        )

    style_x = 790
    elements.extend(
        (
            f'<line x1="{style_x}" y1="52" x2="{style_x + 30}" y2="52" stroke="#374151" stroke-width="2"/>',
            f'<text x="{style_x + 37}" y="57" font-family="sans-serif" font-size="13">train</text>',
            f'<line x1="{style_x + 105}" y1="52" x2="{style_x + 135}" y2="52" stroke="#374151" '
            'stroke-width="2" stroke-dasharray="7 5"/>',
            f'<text x="{style_x + 142}" y="57" font-family="sans-serif" font-size="13">validation</text>',
            "</svg>",
        )
    )
    return "\n".join(elements)


def _default_label(path: Path) -> str:
    log_path = _resolve_log_path(path)
    return log_path.parent.name if log_path.name == "train.jsonl" else log_path.stem


def main() -> None:
    args = parse_args()
    if args.label and len(args.label) != len(args.input):
        raise ValueError("--label must be omitted or repeated exactly once per --input path")
    labels = args.label or [_default_label(path) for path in args.input]
    runs = [load_run(path, label, args.x_axis) for path, label in zip(args.input, labels, strict=True)]
    svg = render_svg(runs, args.title, args.x_axis)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
