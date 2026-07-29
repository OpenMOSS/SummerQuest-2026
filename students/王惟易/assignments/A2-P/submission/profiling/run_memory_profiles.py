import argparse
import json
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
MEMORY_SCRIPT = Path("profiling/memory_snapshot.py")


@dataclass(frozen=True)
class MemoryCase:
    name: str
    model_size: str
    batch_size: int
    context_length: int
    mode: str
    dtype: str


PRIMARY_CASES = tuple(
    MemoryCase(
        name=f"xl-b4-t{context_length}-{mode}-{dtype}",
        model_size="xl",
        batch_size=4,
        context_length=context_length,
        mode=mode,
        dtype=dtype,
    )
    for context_length in (128, 2048)
    for mode in ("forward", "train_step")
    for dtype in ("fp32", "bf16")
)

FALLBACK_CASES = tuple(
    MemoryCase(
        name=f"{model_size}-b1-t{context_length}-train-step-{dtype}",
        model_size=model_size,
        batch_size=1,
        context_length=context_length,
        mode="train_step",
        dtype=dtype,
    )
    for model_size, context_length in (
        ("xl", 2048),
        ("xl", 1024),
        ("large", 2048),
    )
    for dtype in ("fp32", "bf16")
)

MEMORY_CASES = PRIMARY_CASES + FALLBACK_CASES


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run CUDA memory-profile cases in independent processes.",
    )
    parser.add_argument(
        "--suite",
        choices=("primary", "fallback"),
        default="primary",
    )
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in MEMORY_CASES),
        dest="cases",
        help="Run only this case; repeat the option to select multiple cases.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/memory"),
    )
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def resolve_output_dir(path: Path) -> tuple[Path, Path]:
    output_dir = (REPOSITORY_ROOT / path).resolve()
    try:
        relative_output_dir = output_dir.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("--output-dir must be inside the assignment repository") from error
    return output_dir, relative_output_dir


def case_stem(case: MemoryCase) -> str:
    return f"{case.model_size}_b{case.batch_size}_t{case.context_length}_{case.mode}_{case.dtype}"


def child_arguments(
    args: argparse.Namespace,
    case: MemoryCase,
    relative_output_dir: Path,
) -> list[str]:
    stem = case_stem(case)
    return [
        MEMORY_SCRIPT.as_posix(),
        "--model-size",
        case.model_size,
        "--batch-size",
        str(case.batch_size),
        "--context-length",
        str(case.context_length),
        "--mode",
        case.mode,
        "--dtype",
        case.dtype,
        "--warmup",
        str(args.warmup),
        "--max-entries",
        str(args.max_entries),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--snapshot-output",
        (relative_output_dir / f"{stem}.pickle").as_posix(),
        "--metadata-output",
        (relative_output_dir / f"{stem}.metadata.json").as_posix(),
    ]


def main() -> None:
    args = parse_args()
    suite_cases = PRIMARY_CASES if args.suite == "primary" else FALLBACK_CASES
    selected_names = set(args.cases) if args.cases else None
    selected_cases = [case for case in suite_cases if selected_names is None or case.name in selected_names]
    output_dir, relative_output_dir = resolve_output_dir(args.output_dir)

    if selected_names:
        missing = selected_names - {case.name for case in selected_cases}
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"selected cases do not belong to {args.suite} suite: {names}")

    metadata_paths = {case.name: output_dir / f"{case_stem(case)}.metadata.json" for case in selected_cases}
    if not args.dry_run and not args.overwrite:
        existing = [path for path in metadata_paths.values() if path.exists()]
        if existing:
            names = ", ".join(path.name for path in existing)
            raise FileExistsError(
                f"memory metadata already exists: {names}; choose another directory or pass --overwrite",
            )

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    statuses: list[tuple[str, str]] = []
    for case in selected_cases:
        arguments = child_arguments(args, case, relative_output_dir)
        display_command = ["uv", "run", "python", *arguments]
        print(f"$ {shlex.join(display_command)}", flush=True)
        if args.dry_run:
            continue

        completed = subprocess.run(
            [sys.executable, *arguments],
            cwd=REPOSITORY_ROOT,
            check=False,
        )
        metadata_path = metadata_paths[case.name]
        if not metadata_path.exists():
            completed.check_returncode()
            raise FileNotFoundError(
                f"memory case did not write metadata: {metadata_path.name}",
            )

        payload = json.loads(metadata_path.read_text(encoding="utf-8"))
        status = str(payload["status"])
        statuses.append((case.name, status))
        if completed.returncode == 0 and status != "ok":
            raise RuntimeError(
                f"{case.name} exited successfully with unexpected status {status}",
            )
        if completed.returncode != 0 and status != "oom":
            completed.check_returncode()

    if not args.dry_run:
        print("\nMemory case statuses:")
        for name, status in statuses:
            print(f"{name}: {status}")
        print(
            f"\nSummarize with:\n$ uv run python profiling/summarize_memory.py --input-dir {relative_output_dir.as_posix()} --output results/memory.csv",
        )


if __name__ == "__main__":
    main()
