"""Collect Task 3 diagnostic, timing, and peak-memory comparison records."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

from profiling.collect_utils import command_display, failure_kind, require_cuda
from profiling.summarize import read_jsonl, write_benchmark_csv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
MODEL_SIZES = ("small", "medium", "large", "xl", "10B")
MODES = ("forward", "forward_backward")
DTYPES = ("fp32", "bf16")


def append_failure(path: Path, *, model_size: str | None, mode: str | None, dtype: str | None, stage: str, completed: subprocess.CompletedProcess[str]) -> None:
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(
            json.dumps(
                {
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                    "model_size": model_size,
                    "mode": mode,
                    "dtype": dtype,
                    "stage": stage,
                    "exception": failure_kind(completed),
                    "return_code": completed.returncode,
                },
                sort_keys=True,
            )
            + "\n"
        )


def benchmark_command(*, model_size: str, dtype: str, mode: str, output: Path) -> list[str]:
    return [
        sys.executable,
        "profiling/benchmark.py",
        "--model-size",
        model_size,
        "--batch-size",
        "4",
        "--context-length",
        "512",
        "--mode",
        mode,
        "--dtype",
        dtype,
        "--seed",
        "0",
        "--warmup",
        "5",
        "--steps",
        "10",
        "--track-memory",
        "--output",
        str(output),
    ]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect Task 3 accumulation, dtype, FP32, and BF16 data.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--dry-run", action="store_true", help="Print commands without requiring CUDA or writing files.")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    output = args.output_dir / "mixed_precision_benchmark.jsonl"
    diagnostic_output = args.output_dir / "mixed_precision.json"
    toy_command = [sys.executable, "profiling/mixed_precision.py", "toy", "--dtype", "bf16", "--output", str(diagnostic_output)]
    accumulation_command = [sys.executable, "profiling/mixed_precision.py", "accumulation", "--output", str(diagnostic_output)]
    numeric_trend_command = [sys.executable, "profiling/mixed_precision.py", "numeric-trend", "--output", str(diagnostic_output)]
    benchmark_commands = [
        benchmark_command(model_size=model_size, dtype=dtype, mode=mode, output=output)
        for model_size in MODEL_SIZES
        for dtype in DTYPES
        for mode in MODES
    ]
    if args.dry_run:
        for command in (accumulation_command, toy_command, numeric_trend_command, *benchmark_commands):
            print("Planned:", command_display(command), flush=True)
        return
    try:
        require_cuda()
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    failures = args.output_dir / "mixed_precision_failures.jsonl"
    failures.write_text("", encoding="utf-8")

    print("Running:", command_display(accumulation_command), flush=True)
    subprocess.run(accumulation_command, cwd=ROOT, check=True)
    print("Running:", command_display(toy_command), flush=True)
    toy = subprocess.run(toy_command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
    if toy.returncode != 0:
        if toy.stderr:
            print(toy.stderr, file=sys.stderr, end="")
        append_failure(failures, model_size=None, mode=None, dtype="bf16", stage="ToyModel dtype capture", completed=toy)

    print("Running:", command_display(numeric_trend_command), flush=True)
    numeric_trend = subprocess.run(numeric_trend_command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
    if numeric_trend.returncode != 0:
        if numeric_trend.stderr:
            print(numeric_trend.stderr, file=sys.stderr, end="")
        append_failure(
            failures,
            model_size="small",
            mode="train_step",
            dtype="fp32_vs_bf16",
            stage="FP32-versus-BF16 numeric trend",
            completed=numeric_trend,
        )

    for model_size in MODEL_SIZES:
        for dtype in DTYPES:
            for mode in MODES:
                command = benchmark_command(model_size=model_size, dtype=dtype, mode=mode, output=output)
                print("Running:", command_display(command), flush=True)
                completed = subprocess.run(command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
                if completed.returncode != 0:
                    if completed.stderr:
                        print(completed.stderr, file=sys.stderr, end="")
                    append_failure(
                        failures,
                        model_size=model_size,
                        mode=mode,
                        dtype=dtype,
                        stage="benchmark subprocess",
                        completed=completed,
                    )
    write_benchmark_csv(read_jsonl(output), args.output_dir / "mixed_precision_benchmark.csv")


if __name__ == "__main__":
    main()
