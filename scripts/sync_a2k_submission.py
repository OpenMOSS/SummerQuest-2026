#!/usr/bin/env python3
"""Sync A2-K Python files into a SummerQuest submission."""

from __future__ import annotations

import argparse
from pathlib import Path

from a2k_source import copy_submission, validate_source
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
    assignment = root / "students" / name / "assignments" / "A2-K"
    if not (assignment / "README.md").is_file():
        raise FileNotFoundError(
            f"A2-K submission does not exist; run create_assignment.py first: "
            f"{assignment}"
        )
    destination = assignment / "submission"
    copy_submission(source, destination)
    return destination


def main() -> int:
    args = parse_args()
    destination = sync_submission(ROOT, args.name)
    print(
        "Synced A2-K Python allowlist from ../assignment2-systems to "
        f"{destination.relative_to(ROOT)}"
    )
    print(
        "Copied cs336_systems/a2k/**/*.py, tests/adapters.py, and "
        "student_scripts/a2k/**/*.py only."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
