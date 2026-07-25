from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


GROUP_COLUMNS = ("model_size", "mode", "dtype", "batch_size", "context_length", "warmup")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize benchmark CSV files into a Markdown table.")
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("results/timings/summary.md"))
    return parser.parse_args()


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open(newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def markdown_table(rows: list[dict[str, str]]) -> str:
    grouped: dict[tuple[str, ...], list[float]] = defaultdict(list)
    for row in rows:
        key = tuple(row[column] for column in GROUP_COLUMNS)
        grouped[key].append(float(row["seconds"]))

    header = [*GROUP_COLUMNS, "mean_s", "std_s", "n"]
    lines = [
        "| " + " | ".join(header) + " |",
        "| " + " | ".join("---" for _ in header) + " |",
    ]
    for key in sorted(grouped):
        values = grouped[key]
        mean = sum(values) / len(values)
        variance = sum((value - mean) ** 2 for value in values) / (len(values) - 1) if len(values) > 1 else 0.0
        std = variance**0.5
        lines.append("| " + " | ".join([*key, f"{mean:.6f}", f"{std:.6f}", str(len(values))]) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    args = parse_args()
    rows = read_rows(args.inputs)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(markdown_table(rows), encoding="utf-8")
    print(f"Wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
