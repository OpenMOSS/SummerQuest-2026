#!/usr/bin/env python3
"""Sync profiling Python files into a SummerQuest A2-P submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from a2p_source import copy_profiling, validate_source
from create_assignment import validate_name


ROOT = Path(__file__).resolve().parents[1]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="student's real name directory")
    return parser.parse_args()


def sync_submission(root: Path, name: str) -> Path:
    name = name.strip()
    validate_name(name)
    source = validate_source(root)
    assignment = root / "students" / name / "assignments" / "A2-P"
    if not (assignment / "README.md").is_file():
        raise FileNotFoundError(
            f"A2-P submission does not exist; run create_assignment.py first: "
            f"{assignment}"
        )
    destination = assignment / "submission" / "profiling"
    copy_profiling(source, destination)
    return destination


def main() -> int:
    args = parse_args()
    destination = sync_submission(ROOT, args.name)
    print(f"Synced ../assignment2-systems/profiling to {destination.relative_to(ROOT)}")
    print(
        "Only profiling/**/*.py was copied; results, traces, snapshots, and "
        "dependencies were not."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
