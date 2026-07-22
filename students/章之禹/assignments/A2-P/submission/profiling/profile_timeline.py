"""Render a compact, sanitized timeline from a local torch.profiler trace.

The full Chrome trace contains process/thread identifiers and can be tens of
megabytes, so it stays in the private work directory.  This script keeps only
the selected ``record_function`` ranges and a binned GPU-kernel activity strip
for one measured step, then exports a small SVG suitable for the public report.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


STAGES = (
    "forward",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
)


def _events(trace: dict[str, Any]) -> list[dict[str, Any]]:
    events = trace.get("traceEvents", [])
    return [event for event in events if isinstance(event, dict)]


def _overlap(start: float, duration: float, left: float, right: float) -> float:
    return max(0.0, min(start + duration, right) - max(start, left))


def render(trace_path: Path, output: Path, *, title: str, bins: int = 160) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render a timeline") from error

    trace = json.loads(trace_path.read_text(encoding="utf-8"))
    events = _events(trace)
    measurements = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") == "profile/measure"
        and event.get("ph") == "X"
    ]
    if not measurements:
        raise ValueError("trace has no profile/measure annotation")
    measure = measurements[0]
    measure_start = float(measure["ts"])
    measure_end = measure_start + float(measure["dur"])
    measure_width = measure_end - measure_start
    if measure_width <= 0:
        raise ValueError("profile/measure has non-positive duration")

    stage_events = [
        event
        for event in events
        if event.get("cat") == "user_annotation"
        and event.get("name") in STAGES
        and event.get("ph") == "X"
        and measure_start <= float(event.get("ts", -1)) < measure_end
    ]
    kernels = [
        event
        for event in events
        if event.get("cat") == "kernel"
        and event.get("ph") == "X"
        and _overlap(
            float(event.get("ts", 0)),
            float(event.get("dur", 0)),
            measure_start,
            measure_end,
        )
        > 0
    ]

    bin_width = measure_width / bins
    activity = [0.0] * bins
    for kernel in kernels:
        start = float(kernel.get("ts", 0))
        duration = float(kernel.get("dur", 0))
        first = max(0, int((start - measure_start) / bin_width))
        last = min(bins - 1, int((start + duration - measure_start) / bin_width))
        for index in range(first, last + 1):
            left = measure_start + index * bin_width
            right = left + bin_width
            activity[index] += _overlap(start, duration, left, right) / bin_width

    fig, (ranges_axis, kernel_axis) = plt.subplots(
        2,
        1,
        figsize=(9, 5),
        dpi=140,
        sharex=True,
        gridspec_kw={"height_ratios": [3.2, 1]},
    )
    colors = {
        "forward": "#2563eb",
        "backward": "#dc2626",
        "optimizer": "#7c3aed",
        "attention/scores": "#0891b2",
        "attention/softmax": "#0d9488",
        "attention/value": "#65a30d",
    }
    y_positions = {name: len(STAGES) - index for index, name in enumerate(STAGES)}
    for event in stage_events:
        name = str(event["name"])
        start_ms = (float(event["ts"]) - measure_start) / 1000.0
        duration_ms = float(event["dur"]) / 1000.0
        ranges_axis.broken_barh(
            [(start_ms, duration_ms)],
            (y_positions[name] - 0.35, 0.7),
            facecolors=colors[name],
        )
    ranges_axis.set_yticks(
        [y_positions[name] for name in STAGES],
        labels=list(STAGES),
    )
    ranges_axis.set_title(title)
    ranges_axis.set_ylabel("CPU annotation")
    ranges_axis.grid(axis="x", alpha=0.2)

    x_values = [(index + 0.5) * bin_width / 1000.0 for index in range(bins)]
    kernel_axis.fill_between(x_values, activity, color="#f59e0b", alpha=0.75)
    kernel_axis.set_ylabel("kernel\nactivity")
    kernel_axis.set_xlabel("time from profile/measure start (ms)")
    kernel_axis.grid(axis="x", alpha=0.2)
    kernel_axis.set_xlim(0, measure_width / 1000.0)

    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trace", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", default="Representative train_step timeline")
    parser.add_argument("--bins", type=int, default=160)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render(args.trace, args.output, title=args.title, bins=args.bins)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
