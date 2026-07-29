import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
PROFILE_SCRIPT = Path("profiling/compute_profile.py")
ANALYZE_SCRIPT = Path("profiling/analyze_trace.py")


@dataclass(frozen=True)
class ProfileCase:
    model_size: str
    context_length: int

    @property
    def name(self) -> str:
        return f"{self.model_size}-t{self.context_length}"


PROFILE_CASES = tuple(ProfileCase(model_size, context_length) for model_size in ("small", "medium") for context_length in (256, 512, 1024))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the selected torch.profiler configurations in fresh processes.")
    parser.add_argument("--output-dir", type=Path, default=Path("results/profiler/formal"))
    parser.add_argument(
        "--case",
        action="append",
        choices=tuple(case.name for case in PROFILE_CASES),
        dest="cases",
        help="Run only this case; repeat the option to select multiple cases.",
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument(
        "--analyze-only",
        action="store_true",
        help="Reanalyze existing traces without running the model.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace artifacts for selected cases if they already exist.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without creating directories or running profiles.",
    )
    return parser.parse_args()


def resolve_output_dir(path: Path) -> tuple[Path, Path]:
    output_dir = (REPOSITORY_ROOT / path).resolve()
    try:
        relative_output_dir = output_dir.relative_to(REPOSITORY_ROOT)
    except ValueError as error:
        raise ValueError("--output-dir must be inside the assignment repository") from error
    return output_dir, relative_output_dir


def artifact_paths(
    output_dir: Path,
    *,
    case: ProfileCase,
    batch_size: int,
) -> dict[str, Path]:
    stem = f"{case.model_size}_b{batch_size}_t{case.context_length}_fp32"
    return {
        "trace": output_dir / f"{stem}.trace.json",
        "summary": output_dir / f"{stem}.summary.txt",
        "metadata": output_dir / f"{stem}.metadata.json",
        "analysis": output_dir / f"{stem}.analysis.json",
        "log": output_dir / f"{stem}.capture.log",
    }


def relative_paths(paths: dict[str, Path]) -> dict[str, Path]:
    return {name: path.relative_to(REPOSITORY_ROOT) for name, path in paths.items()}


def profile_arguments(
    args: argparse.Namespace,
    case: ProfileCase,
    paths: dict[str, Path],
) -> list[str]:
    return [
        PROFILE_SCRIPT.as_posix(),
        "--model-size",
        case.model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(case.context_length),
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--learning-rate",
        str(args.learning_rate),
        "--trace-output",
        paths["trace"].as_posix(),
        "--summary-output",
        paths["summary"].as_posix(),
        "--metadata-output",
        paths["metadata"].as_posix(),
    ]


def analysis_arguments(
    args: argparse.Namespace,
    paths: dict[str, Path],
) -> list[str]:
    return [
        ANALYZE_SCRIPT.as_posix(),
        "--trace",
        paths["trace"].as_posix(),
        "--output",
        paths["analysis"].as_posix(),
        "--top-k",
        str(args.top_k),
    ]


def main() -> None:
    args = parse_args()
    selected_names = set(args.cases) if args.cases else None
    selected_cases = [case for case in PROFILE_CASES if selected_names is None or case.name in selected_names]
    output_dir, _ = resolve_output_dir(args.output_dir)
    artifacts = {
        case.name: artifact_paths(
            output_dir,
            case=case,
            batch_size=args.batch_size,
        )
        for case in selected_cases
    }

    if not args.dry_run:
        if args.analyze_only:
            missing_traces = [paths["trace"] for paths in artifacts.values() if not paths["trace"].exists()]
            if missing_traces:
                names = ", ".join(path.name for path in missing_traces)
                raise FileNotFoundError(f"profile traces do not exist: {names}")
            if not args.overwrite:
                existing_analyses = [paths["analysis"] for paths in artifacts.values() if paths["analysis"].exists()]
                if existing_analyses:
                    names = ", ".join(path.name for path in existing_analyses)
                    raise FileExistsError(f"profile analyses already exist: {names}; pass --overwrite to replace them")
        elif not args.overwrite:
            existing = [path for paths in artifacts.values() for path in paths.values() if path.exists()]
            if existing:
                names = ", ".join(path.name for path in existing)
                raise FileExistsError(f"profile artifacts already exist: {names}; choose another directory or pass --overwrite")

    if not args.dry_run:
        output_dir.mkdir(parents=True, exist_ok=True)

    for case in selected_cases:
        paths = relative_paths(artifacts[case.name])
        profile_args = profile_arguments(args, case, paths)
        analyze_args = analysis_arguments(args, paths)
        profile_command = ["uv", "run", "python", *profile_args]
        analyze_command = ["uv", "run", "python", *analyze_args]
        if not args.analyze_only:
            print(f"$ {shlex.join(profile_command)}", flush=True)
        print(f"$ {shlex.join(analyze_command)}", flush=True)
        if args.dry_run:
            continue

        if not args.analyze_only:
            with artifacts[case.name]["log"].open("w", encoding="utf-8") as log_file:
                subprocess.run(
                    [sys.executable, *profile_args],
                    cwd=REPOSITORY_ROOT,
                    check=True,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                )
        subprocess.run(
            [sys.executable, *analyze_args],
            cwd=REPOSITORY_ROOT,
            check=True,
        )


if __name__ == "__main__":
    main()
