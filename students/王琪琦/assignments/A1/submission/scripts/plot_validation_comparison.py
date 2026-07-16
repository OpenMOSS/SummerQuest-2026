from __future__ import annotations

import argparse
import html
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare validation-loss JSONL series in SVG.")
    parser.add_argument("--series", action="append", required=True, metavar="LABEL=PATH")
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", required=True)
    args = parser.parse_args()

    series = []
    for specification in args.series:
        label, path = specification.split("=", 1)
        records = [json.loads(line) for line in Path(path).read_text().splitlines() if line]
        values = [(float(row["step"]), float(row["val_loss"])) for row in records if "val_loss" in row]
        if not values:
            raise ValueError(f"series {label!r} has no validation records")
        series.append((label, values))

    width, height = 1040, 600
    left, right, top, bottom = 78, 210, 62, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(step for _, values in series for step, _ in values)
    losses = [loss for _, values in series for _, loss in values]
    y_min = max(0.0, min(losses) - 0.1)
    y_max = min(max(losses) + 0.1, 5.0)
    colors = ["#157f78", "#d4512d", "#315c9b", "#9a6b16", "#7b4d8e", "#50723c"]

    def x_map(value: float) -> float:
        return left + value / max_step * plot_width

    def y_map(value: float) -> float:
        clipped = min(value, y_max)
        return top + (y_max - clipped) / (y_max - y_min) * plot_height

    grid = []
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot_width
        step = round(fraction * max_step)
        y = top + fraction * plot_height
        loss = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>'
            f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle">{step:,}</text>'
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{loss:.2f}</text>'
        )

    lines = []
    legends = []
    for index, (label, values) in enumerate(series):
        color = colors[index % len(colors)]
        coordinates = " ".join(f"{x_map(x):.1f},{y_map(y):.1f}" for x, y in values)
        lines.append(f'<polyline points="{coordinates}" fill="none" stroke="{color}" stroke-width="2.4"/>')
        legend_y = 92 + index * 28
        legends.append(
            f'<line x1="{width - right + 28}" y1="{legend_y}" x2="{width - right + 58}" '
            f'y2="{legend_y}" stroke="{color}" stroke-width="3"/>'
            f'<text x="{width - right + 68}" y="{legend_y + 4}" fill="#172a2a">{html.escape(label)}</text>'
        )

    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f3e8"/>
<text x="{left}" y="36" font-family="Georgia,serif" font-size="24" font-weight="700" fill="#172a2a">{html.escape(args.title)}</text>
<g stroke="#d8d2c2" stroke-width="1" fill="#53605f" font-family="monospace" font-size="11">{''.join(grid)}</g>
{''.join(lines)}
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#172a2a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#172a2a"/>
<text x="{left + plot_width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="Georgia,serif" font-size="14" fill="#172a2a">Training step</text>
<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2:.1f})" font-family="Georgia,serif" font-size="14" fill="#172a2a">Validation loss</text>
<g font-family="monospace" font-size="12">{''.join(legends)}</g>
</svg>'''
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
