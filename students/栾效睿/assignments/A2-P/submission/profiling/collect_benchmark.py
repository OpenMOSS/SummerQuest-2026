"""Collect the minimum Task 1 benchmark matrix required by docs/guide.md."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

from profiling.collect_utils import command_display, require_cuda
from profiling.summarize import read_jsonl, write_benchmark_csv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "benchmark"
RUNS = (("forward", 5), ("forward_backward", 5), ("train_step", 5), ("train_step", 0))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the required Task 1 FP32 baseline matrix.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without requiring CUDA or writing files.")
    return parser


def commands(*, raw_output: Path, steps: int) -> list[list[str]]:
    return [
        [
            sys.executable,
            "profiling/benchmark.py",
            "--model-size",
            "small",
            "--batch-size",
            "4",
            "--context-length",
            "512",
            "--mode",
            mode,
            "--dtype",
            "fp32",
            "--seed",
            "0",
            "--warmup",
            str(warmup),
            "--steps",
            str(steps),
            "--output",
            str(raw_output),
        ]
        for mode, warmup in RUNS
    ]


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error("--steps must be at least 1.")
    raw_output = args.output_dir / "raw.jsonl"
    run_commands = commands(raw_output=raw_output.with_suffix(".jsonl.tmp"), steps=args.steps)
    if args.dry_run:
        for command in run_commands:
            print("Planned:", command_display(command), flush=True)
        return
    try:
        require_cuda()
    except RuntimeError as error:
        parser.error(str(error))
    for command in run_commands:
        print("Running:", command_display(command), flush=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    temporary_raw = raw_output.with_suffix(".jsonl.tmp")
    temporary_csv = (args.output_dir.parent / "benchmark.csv").with_suffix(".csv.tmp")
    temporary_raw.write_text("", encoding="utf-8")
    try:
        for command in run_commands:
            subprocess.run(command, cwd=ROOT, check=True)
        write_benchmark_csv(read_jsonl(temporary_raw), temporary_csv)
        temporary_raw.replace(raw_output)
        temporary_csv.replace(args.output_dir.parent / "benchmark.csv")
    finally:
        temporary_raw.unlink(missing_ok=True)
        temporary_csv.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
