#!/usr/bin/env python3
"""Validate that a student PR changes one student and one review unit."""

from __future__ import annotations

import os
import re
import subprocess
import sys
from pathlib import PurePosixPath


class ScopeError(ValueError):
    """Raised when a pull request crosses its allowed review unit."""


def parse_changed_paths(output: str) -> list[PurePosixPath]:
    """Parse `git diff --name-status -z`, retaining both sides of renames."""
    fields = output.split("\0")
    if fields and not fields[-1]:
        fields.pop()

    paths: list[PurePosixPath] = []
    index = 0
    while index < len(fields):
        status = fields[index]
        index += 1
        if not status:
            continue

        code = status[0]
        path_count = 2 if code in {"R", "C"} else 1
        if index + path_count > len(fields):
            raise RuntimeError(f"malformed git diff --name-status output near {status!r}")

        raw_paths = fields[index : index + path_count]
        index += path_count
        # A rename changes both locations. A copy changes only its destination.
        affected = raw_paths if code == "R" else raw_paths[-1:]
        for raw_path in affected:
            path = PurePosixPath(raw_path)
            if path not in paths:
                paths.append(path)
    return paths


def changed_paths(base: str, head: str) -> list[PurePosixPath]:
    result = subprocess.run(
        [
            "git",
            "diff",
            "--name-status",
            "-z",
            "--find-renames",
            "--diff-filter=ADMRT",
            f"{base}...{head}",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return parse_changed_paths(result.stdout)


def validate_scope(
    paths: list[PurePosixPath], title: str = ""
) -> tuple[str | None, str]:
    student_paths = [
        path
        for path in paths
        if len(path.parts) >= 3
        and path.parts[0] == "students"
        and not path.parts[1].startswith("_")
    ]
    students = {path.parts[1] for path in student_paths}

    if not students:
        return None, "[MAINTAINER]"
    if len(students) != 1:
        raise ScopeError(f"student PR changes multiple student directories: {sorted(students)}")

    student = next(iter(students))
    outside = [path for path in paths if path.parts[:2] != ("students", student)]
    if outside:
        details = "\n".join(f"- {path}" for path in outside)
        raise ScopeError(f"student PR also changes shared or unrelated paths:\n{details}")

    assignments = {
        path.parts[3]
        for path in student_paths
        if len(path.parts) >= 4
        and path.parts[2] == "assignments"
        and re.fullmatch(r"A(?:[0-1]|2-P|[3-6])", path.parts[3])
    }
    if len(assignments) > 1:
        raise ScopeError(f"student PR changes multiple assignments: {sorted(assignments)}")

    profile_path = ("students", student, "PROFILE.md")
    if assignments:
        assignment = next(iter(assignments))
        assignment_prefix = ("students", student, "assignments", assignment)
        if assignment == "A0":
            invalid = [
                path
                for path in student_paths
                if path.parts != profile_path and path.parts[:4] != assignment_prefix
            ]
        else:
            invalid = [
                path for path in student_paths if path.parts[:4] != assignment_prefix
            ]
        expected_label = f"[{assignment}]"
        title_pattern = (
            rf"^{re.escape(expected_label)}(?:\[FIX\])?\s+"
            rf"{re.escape(student)}\s+-\s+\S.*$"
        )
    else:
        invalid = [path for path in student_paths if path.parts != profile_path]
        expected_label = "[PROFILE]"
        title_pattern = (
            rf"^{re.escape(expected_label)}\s+{re.escape(student)}\s+-\s+\S.*$"
        )

    if invalid:
        details = "\n".join(f"- {path}" for path in invalid)
        raise ScopeError(
            f"{expected_label} PR contains files outside its allowed review unit:\n{details}"
        )

    if title and not re.fullmatch(title_pattern, title):
        raise ScopeError(
            f"PR title must match '{expected_label} {student} - <简短说明>': {title}"
        )
    return student, expected_label


def main() -> int:
    if len(sys.argv) != 3:
        print("usage: validate_pr_scope.py <base-sha> <head-sha>", file=sys.stderr)
        return 2

    paths = changed_paths(sys.argv[1], sys.argv[2])
    title = os.environ.get("PR_TITLE", "")
    try:
        student, review_unit = validate_scope(paths, title)
    except ScopeError as exc:
        print(str(exc), file=sys.stderr)
        return 1

    if student is None:
        print("No student submission directory changed; treating this as a maintainer PR.")
        return 0

    print(f"PR scope passed: student={student}, review_unit={review_unit}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
