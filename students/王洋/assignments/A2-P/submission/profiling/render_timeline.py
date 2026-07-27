from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt

CATEGORY_LABELS = (
    "Parameters",
    "Optimizer state",
    "Inputs",
    "Temporary",
    "Activations",
    "Gradients",
    "Autograd detail",
    "Unclassified",
)
CATEGORY_COLORS = (
    "#4e79a7",
    "#f28e2b",
    "#e15759",
    "#76b7b2",
    "#59a14f",
    "#edc949",
    "#af7aa1",
    "#9c755f",
)


def render_timeline(source: Path, output: Path, title: str) -> None:
    times, sizes = json.loads(source.read_text())
    if not times or len(times) != len(sizes):
        raise ValueError("memory timeline must contain aligned timestamp and size arrays")
    elapsed_ms = [(timestamp - times[0]) / 1000 for timestamp in times]
    category_gib = [[row[index] / 2**30 for row in sizes] for index in range(1, len(CATEGORY_LABELS) + 1)]
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(9.5, 4.8))
    axis.stackplot(
        elapsed_ms,
        *category_gib,
        labels=CATEGORY_LABELS,
        colors=CATEGORY_COLORS,
        alpha=0.9,
    )
    axis.set_xlabel("Profiler timeline (ms)")
    axis.set_ylabel("Active tensor memory (GiB)")
    axis.set_title(title)
    axis.legend(loc="upper left", ncol=2, fontsize=7)
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a PyTorch memory timeline JSON as a PNG.")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--title", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    render_timeline(args.source, args.output, args.title)


if __name__ == "__main__":
    main()
