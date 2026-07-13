#!/usr/bin/env python3
"""Create one A1-A6 student assignment directory from the public template."""

from __future__ import annotations

import argparse
import re
import shutil
from pathlib import Path

from a1_source import copy_submission, validate_source


ROOT = Path(__file__).resolve().parents[1]
ASSIGNMENT = re.compile(r"A[1-6]")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True, help="student's real name directory")
    parser.add_argument("--assignment", required=True, help="assignment number, A1-A6")
    return parser.parse_args()


def validate_name(name: str) -> None:
    if not name or name.startswith("_") or any(char.isspace() for char in name):
        raise ValueError(
            "--name must be a real name without spaces and cannot start with '_'"
        )
    if any(char in name for char in ("/", "\\")):
        raise ValueError("--name cannot contain path separators")


def create_assignment(root: Path, name: str, assignment: str) -> Path:
    name = name.strip()
    assignment = assignment.strip().upper()
    validate_name(name)
    if not ASSIGNMENT.fullmatch(assignment):
        raise ValueError(
            "--assignment must be one of A1-A6; A0 is created by create_student.py"
        )

    student = root / "students" / name
    if not (student / "PROFILE.md").is_file():
        raise FileNotFoundError(
            f"student directory does not exist or has no PROFILE.md: {student}"
        )

    generic_template = root / "students" / "_assignment_template"
    template = (
        root / "students" / "_assignment_templates" / "A1"
        if assignment == "A1"
        else generic_template
    )
    template_report = template / "README.md"
    if not template_report.is_file():
        raise FileNotFoundError(f"assignment template is missing: {template_report}")

    a1_source: Path | None = None
    if assignment == "A1":
        a1_source = validate_source(root)

    destination = student / "assignments" / assignment
    if destination.exists():
        raise FileExistsError(f"assignment directory already exists: {destination}")

    shutil.copytree(template, destination)

    if assignment == "A1":
        assert a1_source is not None
        submission = destination / "submission"
        copy_submission(a1_source, submission)
        (destination / "logs").mkdir()
        (destination / "assets").mkdir()

    replacements = {
        "<姓名>": name,
        "<同学真名>": name,
        "<A编号>": assignment,
    }
    for path in destination.rglob("*.md"):
        text = path.read_text(encoding="utf-8")
        for source, target in replacements.items():
            text = text.replace(source, target)
        path.write_text(text, encoding="utf-8")
    return destination


def main() -> int:
    args = parse_args()
    destination = create_assignment(ROOT, args.name, args.assignment)
    print(f"Created {destination.relative_to(ROOT)}")
    print(
        "Next: follow the formal assignment handout, fill every remaining "
        "<...> placeholder, and run python scripts/validate_repo.py."
    )
    if args.assignment.strip().upper() == "A1":
        print(
            "A1 workspace: ../assignment1-basics. Work and test there, then run "
            "python3 scripts/sync_a1_submission.py --name '<同学真名>'."
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
