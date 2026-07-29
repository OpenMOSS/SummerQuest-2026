"""Run the unmodified official attention tests and save a public-safe transcript."""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import pytest

from student_scripts.a2k.common import (
    configure_single_gpu,
    public_gpu_metadata,
)

STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--uv",
        action="store_true",
        help="execute the official tests through uv in a fresh Python process",
    )
    return parser.parse_args()


def run_with_uv() -> tuple[int, str, str]:
    command = [
        "uv",
        "run",
        "--active",
        "--no-sync",
        "pytest",
        "tests/test_attention.py",
        "-v",
    ]
    with tempfile.TemporaryDirectory(prefix="a2k-pytest-guard-") as temporary:
        helper = Path(temporary) / "sitecustomize.py"
        helper.write_text(
            "import torch\n"
            "total = torch.cuda.get_device_properties(0).total_memory\n"
            "limit = 23 * 1024**3\n"
            "torch.cuda.set_per_process_memory_fraction("
            "min(1.0, limit / total), device=0)\n",
            encoding="utf-8",
        )
        environment = os.environ.copy()
        existing_pythonpath = environment.get("PYTHONPATH", "")
        environment["PYTHONPATH"] = (
            temporary
            if not existing_pythonpath
            else f"{temporary}{os.pathsep}{existing_pythonpath}"
        )
        result = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            env=environment,
        )
    return result.returncode, result.stdout + result.stderr, " ".join(command)


def main() -> int:
    args = parse_args()
    configure_single_gpu()
    gpu = public_gpu_metadata()
    if args.uv:
        exit_code, text, executed_command = run_with_uv()
    else:
        transcript = io.StringIO()
        with contextlib.redirect_stdout(transcript), contextlib.redirect_stderr(transcript):
            exit_code = pytest.main(["tests/test_attention.py", "-v"])
        text = transcript.getvalue()
        executed_command = "python -m pytest tests/test_attention.py -v"
    for sensitive, replacement in (
        (str(Path.cwd()), "<upstream-workspace>"),
        (sys.executable, "<python>"),
    ):
        text = text.replace(sensitive, replacement)
    text = re.sub(
        r"(?m)(platform .+ -- )/[^\s]+/python(?:[0-9.]*)?$",
        r"\1<python>",
        text,
    )
    # The pytest output contains only public test names and software behavior.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        "Wrapper command: python -m student_scripts.a2k.run_unit_tests "
        "--uv --output results/unit_tests.txt\n"
        f"Executed tests: {executed_command}\n"
        f"GPU: {gpu['gpu_name']}\n"
        f"Starter commit: {STARTER_COMMIT}\n"
        "Allocator limit: 23552 MiB\n\n" + text,
        encoding="utf-8",
    )
    print(text)
    return int(exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
