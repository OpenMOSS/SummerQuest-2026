import argparse
import html
import json
import math
from pathlib import Path


COLORS = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2", "#4f46e5"]


def load_series(specification, x_key, metric):
    label, separator, path = specification.partition("=")
    if not separator:
        raise ValueError(f"series must have LABEL=PATH form: {specification}")
    points = []
    with open(path, encoding="utf-8") as source:
        for line in source:
            record = json.loads(line)
            x, y = record.get(x_key), record.get(metric)
            if x is not None and y is not None and math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
    if not points:
        raise ValueError(f"no finite {metric} points in {path}")
    return label, points


def main() -> None:
    parser = argparse.ArgumentParser(description="Render JSONL loss logs as a dependency-free SVG")
    parser.add_argument("--series", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--x", choices=("step", "wall_clock_sec"), default="step")
    parser.add_argument("--metric", choices=("train_loss", "val_loss"), default="val_loss")
    parser.add_argument("--title", default="Validation loss")
    args = parser.parse_args()

    series = [load_series(item, args.x, args.metric) for item in args.series]
    all_points = [point for _, points in series for point in points]
    min_x, max_x = min(x for x, _ in all_points), max(x for x, _ in all_points)
    min_y, max_y = min(y for _, y in all_points), max(y for _, y in all_points)
    if min_x == max_x:
        max_x += 1
    if min_y == max_y:
        max_y += 1
    y_padding = (max_y - min_y) * 0.05
    min_y, max_y = max(0, min_y - y_padding), max_y + y_padding

    width, height = 900, 520
    left, right, top, bottom = 80, 25, 55, 70
    plot_width, plot_height = width - left - right, height - top - bottom

    def sx(value):
        return left + (value - min_x) / (max_x - min_x) * plot_width

    def sy(value):
        return top + (max_y - value) / (max_y - min_y) * plot_height

    elements = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        f'<text x="{width / 2}" y="30" text-anchor="middle" font-family="sans-serif" font-size="20">'
        f"{html.escape(args.title)}</text>",
    ]
    for tick in range(6):
        fraction = tick / 5
        x_value = min_x + fraction * (max_x - min_x)
        x = sx(x_value)
        elements.append(f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}" stroke="#e5e7eb"/>')
        elements.append(
            f'<text x="{x:.1f}" y="{top + plot_height + 24}" text-anchor="middle" '
            f'font-family="sans-serif" font-size="12">{x_value:.0f}</text>'
        )
        y_value = max_y - fraction * (max_y - min_y)
        y = sy(y_value)
        elements.append(f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}" stroke="#e5e7eb"/>')
        elements.append(
            f'<text x="{left - 10}" y="{y + 4:.1f}" text-anchor="end" '
            f'font-family="sans-serif" font-size="12">{y_value:.2f}</text>'
        )
    elements.extend((
        f'<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#111827"/>',
        f'<text x="{left + plot_width / 2}" y="{height - 20}" text-anchor="middle" '
        f'font-family="sans-serif" font-size="14">{html.escape(args.x)}</text>',
        f'<text x="18" y="{top + plot_height / 2}" text-anchor="middle" transform="rotate(-90 18 '
        f'{top + plot_height / 2})" font-family="sans-serif" font-size="14">{html.escape(args.metric)}</text>',
    ))
    for index, (label, points) in enumerate(series):
        color = COLORS[index % len(COLORS)]
        coordinates = " ".join(f"{sx(x):.1f},{sy(y):.1f}" for x, y in points)
        elements.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2"/>')
        legend_x, legend_y = left + 12 + (index % 4) * 190, top + 18 + (index // 4) * 22
        elements.append(f'<line x1="{legend_x}" y1="{legend_y}" x2="{legend_x + 24}" y2="{legend_y}" stroke="{color}" stroke-width="3"/>')
        elements.append(
            f'<text x="{legend_x + 30}" y="{legend_y + 4}" font-family="sans-serif" font-size="12">'
            f"{html.escape(label)}</text>"
        )
    elements.append("</svg>")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(elements), encoding="utf-8")


if __name__ == "__main__":
    main()
