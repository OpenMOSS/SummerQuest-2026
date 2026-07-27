#!/usr/bin/env python3
"""Summarize torch.profiler exports into a light CSV table."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def summarize(path: Path, output: Path) -> None:
    rows = json.loads(path.read_text())
    if isinstance(rows, dict):
        rows = rows.get("rows", [])
    fields = sorted({key for row in rows for key in row})
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    summarize(args.input, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
