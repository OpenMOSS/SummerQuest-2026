from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


SENSITIVE_PATTERNS = [
    re.compile(r"/(?:home|inspire)/[^\s]+"),
    re.compile(r"GPU-[0-9A-Fa-f-]+"),
    re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b"),
]


def sanitize(text: str) -> str:
    for pattern in SENSITIVE_PATTERNS:
        text = pattern.sub("<redacted>", text)
    return text


def run(cmd: list[str], *, cwd: Path, output: Path | None = None) -> None:
    env = os.environ.copy()
    env.setdefault("PYTHONHASHSEED", "0")
    proc = subprocess.run(cmd, cwd=cwd, env=env, text=True, capture_output=True)
    if output is not None:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(sanitize(proc.stdout + proc.stderr), encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--assets-dir", type=Path, default=Path("local_results/a2k/assets"))
    parser.add_argument("--python", default=sys.executable)
    args = parser.parse_args()
    repo = Path.cwd()
    results_dir = args.results_dir
    assets_dir = args.assets_dir

    run([args.python, "-m", "pytest", "tests/test_attention.py", "-v"], cwd=repo, output=results_dir / "unit_tests.txt")
    run([args.python, "-m", "student_scripts.a2k.correctness", "--include-triton", "--output", str(results_dir / "correctness.json")], cwd=repo)
    run([args.python, "-m", "student_scripts.a2k.benchmark_checkpointing", "--output", str(results_dir / "checkpointing.csv")], cwd=repo)
    run([args.python, "-m", "student_scripts.a2k.benchmark_attention", "--output-dir", str(results_dir)], cwd=repo)
    run([args.python, "-m", "student_scripts.a2k.collect_metadata", "--output", str(results_dir / "run_metadata.json")], cwd=repo)
    run([args.python, "-m", "student_scripts.a2k.memory_evidence", "--results-dir", str(results_dir), "--output", str(results_dir / "memory_evidence.json")], cwd=repo)
    run([args.python, "-m", "student_scripts.a2k.plot_results", "--results-dir", str(results_dir), "--assets-dir", str(assets_dir)], cwd=repo)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
