from __future__ import annotations

import argparse
import html
import json
from collections.abc import Callable
from pathlib import Path


def points(
    values: list[tuple[float, float]],
    x_map: Callable[[float], float],
    y_map: Callable[[float], float],
) -> str:
    return " ".join(f"{x_map(x):.1f},{y_map(y):.1f}" for x, y in values)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render a training JSONL loss curve as SVG.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--title", default="Training loss")
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.input).read_text().splitlines() if line]
    train = [(float(record["step"]), float(record["train_loss"])) for record in records]
    valid = [
        (float(record["step"]), float(record["val_loss"]))
        for record in records
        if "val_loss" in record
    ]
    if not train:
        raise ValueError("input log contains no training records")

    width, height = 960, 540
    left, right, top, bottom = 78, 28, 62, 64
    plot_width = width - left - right
    plot_height = height - top - bottom
    max_step = max(step for step, _ in train)
    losses = [loss for _, loss in train] + [loss for _, loss in valid]
    y_min = max(0.0, min(losses) - 0.25)
    y_max = max(losses) + 0.25
    def x_map(value: float) -> float:
        return left + value / max_step * plot_width

    def y_map(value: float) -> float:
        return top + (y_max - value) / (y_max - y_min) * plot_height

    grid = []
    for index in range(6):
        fraction = index / 5
        x = left + fraction * plot_width
        step = round(fraction * max_step)
        grid.append(
            f'<line x1="{x:.1f}" y1="{top}" x2="{x:.1f}" y2="{top + plot_height}"/>'
            f'<text x="{x:.1f}" y="{height - 30}" text-anchor="middle">{step:,}</text>'
        )
        y = top + fraction * plot_height
        loss = y_max - fraction * (y_max - y_min)
        grid.append(
            f'<line x1="{left}" y1="{y:.1f}" x2="{left + plot_width}" y2="{y:.1f}"/>'
            f'<text x="{left - 12}" y="{y + 4:.1f}" text-anchor="end">{loss:.2f}</text>'
        )

    validation_dots = "".join(
        f'<circle cx="{x_map(step):.1f}" cy="{y_map(loss):.1f}" r="3.4"/>'
        for step, loss in valid
    )
    svg = f'''<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
<rect width="100%" height="100%" fill="#f7f3e8"/>
<text x="{left}" y="36" font-family="Georgia,serif" font-size="24" font-weight="700" fill="#172a2a">{html.escape(args.title)}</text>
<g stroke="#d8d2c2" stroke-width="1" fill="#53605f" font-family="monospace" font-size="11">{''.join(grid)}</g>
<polyline points="{points(train, x_map, y_map)}" fill="none" stroke="#157f78" stroke-width="2" stroke-opacity="0.78"/>
<polyline points="{points(valid, x_map, y_map)}" fill="none" stroke="#d4512d" stroke-width="2.6"/>
<g fill="#d4512d">{validation_dots}</g>
<line x1="{left}" y1="{top + plot_height}" x2="{left + plot_width}" y2="{top + plot_height}" stroke="#172a2a"/>
<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_height}" stroke="#172a2a"/>
<text x="{left + plot_width / 2:.1f}" y="{height - 8}" text-anchor="middle" font-family="Georgia,serif" font-size="14" fill="#172a2a">Training step</text>
<text x="18" y="{top + plot_height / 2:.1f}" text-anchor="middle" transform="rotate(-90 18 {top + plot_height / 2:.1f})" font-family="Georgia,serif" font-size="14" fill="#172a2a">Cross-entropy loss</text>
<g font-family="monospace" font-size="12"><line x1="690" y1="27" x2="718" y2="27" stroke="#157f78" stroke-width="2"/><text x="726" y="31" fill="#172a2a">train</text><line x1="790" y1="27" x2="818" y2="27" stroke="#d4512d" stroke-width="2.6"/><text x="826" y="31" fill="#172a2a">validation</text></g>
</svg>'''
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(svg, encoding="utf-8")


if __name__ == "__main__":
    main()
