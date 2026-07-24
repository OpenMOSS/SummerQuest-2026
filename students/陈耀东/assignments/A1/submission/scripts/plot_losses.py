"""把一个或多个 JSONL 指标文件绘制为无额外依赖的 SVG 曲线。"""

from __future__ import annotations

import argparse
import json
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path


WIDTH = 960
HEIGHT = 600
LEFT = 84
RIGHT = 32
TOP = 58
BOTTOM = 72
COLORS = ("#1f77b4", "#d62728", "#2ca02c", "#9467bd", "#8c564b", "#17becf")


@dataclass(frozen=True)
class Series:
    """一条待绘制的实验曲线。"""

    label: str
    points: list[tuple[float, float]]


def parse_args() -> argparse.Namespace:
    """解析曲线参数。"""
    parser = argparse.ArgumentParser(description="从 CS336 JSONL 日志生成 SVG loss 曲线")
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSONL",
        help="可重复传入多条实验曲线",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metric", choices=("train_loss", "val_loss"), default="val_loss")
    parser.add_argument("--x-axis", choices=("step", "wall_clock_sec"), default="step")
    parser.add_argument("--title", default="CS336 Loss Curves")
    return parser.parse_args()


def parse_run_spec(spec: str) -> tuple[str, Path]:
    """解析 LABEL=PATH 格式。"""
    if "=" not in spec:
        raise ValueError("--run 必须使用 LABEL=METRICS_JSONL 格式")
    label, path_text = spec.split("=", maxsplit=1)
    if not label.strip() or not path_text.strip():
        raise ValueError("--run 的 label 和 path 都不能为空")
    return label.strip(), Path(path_text.strip())


def load_series(label: str, path: Path, x_key: str, y_key: str) -> Series:
    """从 JSONL 中提取有限坐标点。"""
    points: list[tuple[float, float]] = []
    with path.open(encoding="utf-8") as metrics_file:
        for line in metrics_file:
            if not line.strip():
                continue
            record = json.loads(line)
            x_value = record.get(x_key)
            y_value = record.get(y_key)
            if not isinstance(x_value, (int, float)) or not isinstance(y_value, (int, float)):
                continue
            x = float(x_value)
            y = float(y_value)
            if math.isfinite(x) and math.isfinite(y):
                points.append((x, y))
    if not points:
        raise ValueError(f"{path} 没有可绘制的 {y_key} 数据")
    return Series(label=label, points=points)


def svg_element(tag: str, **attributes: object) -> ET.Element:
    """创建属性统一转为字符串的 SVG 元素。"""
    return ET.Element(tag, {key.replace("_", "-"): str(value) for key, value in attributes.items()})


def add_text(parent: ET.Element, x: float, y: float, content: str, **attributes: object) -> None:
    """向 SVG 添加文本。"""
    text_attributes = {
        "x": f"{x:.2f}",
        "y": f"{y:.2f}",
        **{key.replace("_", "-"): str(value) for key, value in attributes.items()},
    }
    element = ET.SubElement(parent, "text", text_attributes)
    element.text = content


def padded_range(values: list[float]) -> tuple[float, float]:
    """给坐标范围增加边距，并处理所有值相同的情况。"""
    minimum = min(values)
    maximum = max(values)
    if minimum == maximum:
        padding = max(abs(minimum) * 0.05, 1.0)
    else:
        padding = (maximum - minimum) * 0.05
    return minimum - padding, maximum + padding


def render_svg(series_list: list[Series], x_label: str, y_label: str, title: str) -> ET.ElementTree:
    """把多条曲线渲染为 SVG DOM。"""
    all_x = [x for series in series_list for x, _ in series.points]
    all_y = [y for series in series_list for _, y in series.points]
    x_min, x_max = padded_range(all_x)
    y_min, y_max = padded_range(all_y)
    plot_width = WIDTH - LEFT - RIGHT
    plot_height = HEIGHT - TOP - BOTTOM

    def map_x(value: float) -> float:
        return LEFT + (value - x_min) / (x_max - x_min) * plot_width

    def map_y(value: float) -> float:
        return TOP + (y_max - value) / (y_max - y_min) * plot_height

    root = svg_element(
        "svg",
        xmlns="http://www.w3.org/2000/svg",
        viewBox=f"0 0 {WIDTH} {HEIGHT}",
        width=WIDTH,
        height=HEIGHT,
        role="img",
    )
    root.append(svg_element("rect", x=0, y=0, width=WIDTH, height=HEIGHT, fill="#ffffff"))
    add_text(root, WIDTH / 2, 32, title, text_anchor="middle", font_size=22, font_family="Arial, sans-serif")

    for index in range(6):
        fraction = index / 5
        x = LEFT + fraction * plot_width
        y = TOP + fraction * plot_height
        root.append(svg_element("line", x1=x, y1=TOP, x2=x, y2=TOP + plot_height, stroke="#e5e7eb"))
        root.append(svg_element("line", x1=LEFT, y1=y, x2=LEFT + plot_width, y2=y, stroke="#e5e7eb"))
        x_value = x_min + fraction * (x_max - x_min)
        y_value = y_max - fraction * (y_max - y_min)
        add_text(root, x, TOP + plot_height + 24, f"{x_value:.3g}", text_anchor="middle", font_size=12)
        add_text(root, LEFT - 12, y + 4, f"{y_value:.3g}", text_anchor="end", font_size=12)

    root.append(svg_element("line", x1=LEFT, y1=TOP, x2=LEFT, y2=TOP + plot_height, stroke="#111827", stroke_width=2))
    root.append(svg_element("line", x1=LEFT, y1=TOP + plot_height, x2=LEFT + plot_width, y2=TOP + plot_height, stroke="#111827", stroke_width=2))
    add_text(root, LEFT + plot_width / 2, HEIGHT - 20, x_label, text_anchor="middle", font_size=14)
    add_text(
        root,
        22,
        TOP + plot_height / 2,
        y_label,
        text_anchor="middle",
        font_size=14,
        transform=f"rotate(-90 22 {TOP + plot_height / 2:.2f})",
    )

    for index, series in enumerate(series_list):
        color = COLORS[index % len(COLORS)]
        coordinates = [(map_x(x), map_y(y)) for x, y in series.points]
        path_data = " ".join(
            f"{'M' if point_index == 0 else 'L'} {x:.2f} {y:.2f}"
            for point_index, (x, y) in enumerate(coordinates)
        )
        root.append(svg_element("path", d=path_data, fill="none", stroke=color, stroke_width=2.5))
        for x, y in coordinates:
            root.append(svg_element("circle", cx=x, cy=y, r=2.8, fill=color))

        legend_x = LEFT + 12 + (index % 3) * 250
        legend_y = TOP + 20 + (index // 3) * 24
        root.append(svg_element("line", x1=legend_x, y1=legend_y, x2=legend_x + 28, y2=legend_y, stroke=color, stroke_width=3))
        add_text(root, legend_x + 36, legend_y + 4, series.label, font_size=13)

    return ET.ElementTree(root)


def main() -> None:
    """加载曲线、渲染 SVG 并写出文件。"""
    args = parse_args()
    series_list = [
        load_series(label, path, args.x_axis, args.metric)
        for label, path in (parse_run_spec(spec) for spec in args.run)
    ]
    document = render_svg(
        series_list=series_list,
        x_label=args.x_axis,
        y_label=args.metric,
        title=args.title,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(document, space="  ")
    document.write(args.output, encoding="utf-8", xml_declaration=True)
    print(f"series={len(series_list)} output={args.output}")


if __name__ == "__main__":
    main()
