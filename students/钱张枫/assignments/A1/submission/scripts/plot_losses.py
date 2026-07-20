from __future__ import annotations

import argparse

from cs336_basics.Part5.plotting import CurveSeries, render_loss_svg
from cs336_basics.Part5.training import resolve_project_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Render one or more JSONL loss curves as a portable SVG.")
    parser.add_argument(
        "--series",
        action="append",
        required=True,
        metavar="LABEL=METRICS_JSONL",
        help="Repeat this option to compare multiple experiment logs.",
    )
    parser.add_argument("--metric", choices=("train_loss", "val_loss"), default="val_loss")
    parser.add_argument("--x", choices=("step", "wall_clock_sec", "processed_tokens"), default="step")
    parser.add_argument("--title", default="Loss curve")
    parser.add_argument("--output", required=True, help="Destination .svg path.")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    series_list = [_parse_series(value, args.metric) for value in args.series]
    output_path = render_loss_svg(
        series_list,
        resolve_project_path(args.output),
        x_key=args.x,
        title=args.title,
    )
    print(output_path)
    return 0


def _parse_series(value: str, metric_key: str) -> CurveSeries:
    label, separator, path = value.partition("=")
    if not separator or not label.strip() or not path.strip():
        raise ValueError("--series must use the form LABEL=METRICS_JSONL.")
    metrics_path = resolve_project_path(path)
    if not metrics_path.is_file():
        raise FileNotFoundError(f"metrics file does not exist: {metrics_path}")
    return CurveSeries(label=label.strip(), metrics_path=metrics_path, metric_key=metric_key)


if __name__ == "__main__":
    raise SystemExit(main())
