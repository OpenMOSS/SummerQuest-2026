"""Run official attention tests and save a path-sanitized text record."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
from pathlib import Path

import pytest

from student_scripts.a2k.common import configure_cuda_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    configure_cuda_environment(require_rtx4090=True)
    repository = Path(__file__).resolve().parents[2]
    os.chdir(repository)
    captured = io.StringIO()
    with contextlib.redirect_stdout(captured), contextlib.redirect_stderr(captured):
        exit_code = pytest.main(["tests/test_attention.py", "-v"])
    combined = captured.getvalue()
    sanitized = combined.replace(str(repository), "[REPOSITORY]")
    header = (
        "Command: python -m pytest tests/test_attention.py -v\n"
        "GPU requirement: RTX 4090 24GB\n"
        f"Exit code: {int(exit_code)}\n\n"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "unit_tests.txt"
    output_path.write_text(header + sanitized, encoding="utf-8")
    print(f"saved official test output to {output_path}")
    raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()
