#!/usr/bin/env python3
"""Run one benchmark and write a CUDA memory-history snapshot."""

from __future__ import annotations

import argparse
from pathlib import Path

from scripts.benchmark import main as benchmark_main


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args, remainder = parser.parse_known_args()
    return benchmark_main(["--memory-snapshot", str(args.output), *remainder])


if __name__ == "__main__":
    raise SystemExit(main())
