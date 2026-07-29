import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
BENCHMARK_SCRIPT = Path("profiling/benchmark.py")


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    mode: str
    warmup_steps: int


BENCHMARK_CASES = (
    BenchmarkCase("forward-w5", "forward", 5),
    BenchmarkCase("forward-backward-w5", "forward_backward", 5),
    BenchmarkCase("train-step-w5", "train_step", 5),
    BenchmarkCase("train-step-w0", "train_step", 0),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the required end-to-end benchmarks in independent processes.",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("results/benchmark/raw"))
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in BENCHMARK_CASES),
        dest="cases",
        help="Run only this case; repeat the option to select multiple cases.",
    )
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument(
        "--dtype",
        choices=(
            "fp32",
            "bf16",
        ),
        default="fp32",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace JSON files for selected cases if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating directories or running benchmarks.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> tuple[Path, Path]:
    output_dir = (REPOSITORY_ROOT / path).resolve()
    try:
        relative_output_dir = output_dir.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("--output-dir must be inside the assignment repository") from error
    return output_dir, relative_output_dir


def output_name(args: argparse.Namespace, case: BenchmarkCase) -> str:
    return f"{args.model_size}_b{args.batch_size}_t{args.context_length}_{case.mode}_{args.dtype}_w{case.warmup_steps}.json"


def benchmark_arguments(
    args: argparse.Namespace,
    case: BenchmarkCase,
    relative_output: Path,
) -> list[str]:
    return [
        BENCHMARK_SCRIPT.as_posix(),
        "--model-size",
        args.model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--mode",
        case.mode,
        "--warmup",
        str(case.warmup_steps),
        "--steps",
        str(args.steps),
        "--dtype",
        args.dtype,
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--device",
        args.device,
        "--output",
        relative_output.as_posix(),
    ]


def main() -> None:
    args = parse_args()
    selected_names = set(args.cases) if args.cases else None
    selected_cases = [case for case in BENCHMARK_CASES if selected_names is None or case.name in selected_names]
    output_dir, relative_output_dir = resolve_output_dir(args.output_dir)
    outputs = {case.name: output_dir / output_name(args, case) for case in selected_cases}

    if not args.dry_run and not args.overwrite:
        existing = [path for path in outputs.values() if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(f"benchmark output already exists: {names}; choose another directory or pass --overwrite")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for case in selected_cases:
        relative_output = relative_output_dir / outputs[case.name].name
        child_arguments = benchmark_arguments(args, case, relative_output)
        display_command = ["uv", "run", "python", *child_arguments]
        print(f"$ {shlex.join(display_command)}", flush=True)
        if not args.dry_run:
            subprocess.run(
                [sys.executable, *child_arguments],
                cwd=REPOSITORY_ROOT,
                check=True,
            )

    if not args.dry_run:
        print(f"\nSummarize with:\n$ uv run python profiling/summarize.py --input-dir {relative_output_dir.as_posix()} --output results/benchmark.csv")


if __name__ == "__main__":
    main()
