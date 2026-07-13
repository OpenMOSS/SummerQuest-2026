#!/usr/bin/env python3
"""Sync selected student work from ../assignment1-basics into a SummerQuest A1 submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from a1_source import copy_submission, validate_source
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
    assignment = root / "students" / name / "assignments" / "A1"
    if not (assignment / "README.md").is_file():
        raise FileNotFoundError(
            f"A1 submission does not exist; run create_assignment.py first: {assignment}"
        )
    submission = assignment / "submission"
    copy_submission(source, submission)
    return submission


def main() -> int:
    args = parse_args()
    destination = sync_submission(ROOT, args.name)
    print(f"Synced ../assignment1-basics to {destination.relative_to(ROOT)}")
    print("Only cs336_basics/, tests/adapters.py, scripts/, and configs/ were copied.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
