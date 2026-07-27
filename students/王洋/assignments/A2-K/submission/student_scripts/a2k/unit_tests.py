from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path

import pytest

from student_scripts.a2k.common import configure_formal_process, public_environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the official A2-K attention tests under the 23 GiB guard.")
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = configure_formal_process()
    environment = public_environment(fraction)
    stream = io.StringIO()
    pytest_args = [
        "tests/test_attention.py",
        "-v",
        "--tb=short",
        "--disable-warnings",
        "-rA",
    ]
    with contextlib.redirect_stdout(stream), contextlib.redirect_stderr(stream):
        exit_code = pytest.main(pytest_args)
    output = stream.getvalue().replace(str(Path.cwd().resolve()), ".")
    header = {
        "command": "pytest tests/test_attention.py -v --tb=short --disable-warnings -rA",
        "allocator_guard_applied_before_tests": True,
        "environment": environment,
        "pytest_exit_code": int(exit_code),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(header, indent=2) + "\n\n" + output)
    print(output, end="")
    if exit_code != pytest.ExitCode.OK:
        raise SystemExit(int(exit_code))


if __name__ == "__main__":
    main()
