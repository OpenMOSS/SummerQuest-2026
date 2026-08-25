from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path

CATEGORIES = ("tests", "correctness", "checkpoint", "compile", "attention")


def sanitize_public_test_output(text: str) -> str:
    text = re.sub(
        r"(?m)(-- )/\S+/bin/python$",
        r"\1python",
        text,
    )
    return re.sub(
        r"(?m)^rootdir: .+$",
        "rootdir: assignment2-systems",
        text,
    )


def run_child(
    command: list[str],
    *,
    log_path: Path,
    expected_output: Path | None = None,
    rerun: bool = False,
) -> int:
    if expected_output is not None and expected_output.exists() and not rerun:
        print(f"skip: {expected_output}")
        return 0
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("run:", " ".join(command))
    completed = subprocess.run(
        command,
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "PYTHONUNBUFFERED": "1",
        },
    )
    log_path.write_text(completed.stdout + "\n--- stderr ---\n" + completed.stderr)
    if completed.returncode != 0:
        print(f"child exited {completed.returncode}: {log_path}")
    return completed.returncode


def common_flags(args: argparse.Namespace) -> list[str]:
    flags = []
    if args.allow_nonstandard_gpu:
        flags.append("--allow-nonstandard-gpu")
    if args.allow_low_free_memory:
        flags.append("--allow-low-free-memory")
    return flags


def run_tests(root: Path, args: argparse.Namespace) -> None:
    output = root / "results" / "unit_tests.txt"
    if output.exists() and not args.rerun:
        print(f"skip: {output}")
        return
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            "tests/test_attention.py",
            "-v",
        ],
        text=True,
        capture_output=True,
        env={
            **os.environ,
            "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        },
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        sanitize_public_test_output(
            completed.stdout + "\n--- stderr ---\n" + completed.stderr
        )
    )
    if completed.returncode != 0:
        print(f"attention tests failed: {output}")


def run_correctness(root: Path, args: argparse.Namespace) -> None:
    output = root / "results" / "correctness.json"
    run_child(
        [
            sys.executable,
            "student_scripts/a2k/correctness.py",
            "--output",
            str(output),
            *common_flags(args),
        ],
        log_path=root / "logs" / "correctness.log",
        expected_output=output,
        rerun=args.rerun,
    )


def checkpoint_output(root: Path, context: int, block: int) -> Path:
    return root / "raw" / "checkpoint" / f"s{context}_ckpt{block}.json"


def run_checkpoint(root: Path, args: argparse.Namespace) -> None:
    for block in (0, 1, 2, 4, 8):
        output = checkpoint_output(root, 1024, block)
        run_child(
            [
                sys.executable,
                "student_scripts/a2k/checkpoint_benchmark.py",
                "--context-length",
                "1024",
                "--checkpoint-block-size",
                str(block),
                "--output",
                str(output),
                *common_flags(args),
            ],
            log_path=root / "logs" / "checkpoint" / f"s1024_ckpt{block}.log",
            expected_output=output,
            rerun=args.rerun,
        )

    successful = []
    for block in (1, 2, 4, 8):
        path = checkpoint_output(root, 1024, block)
        if path.exists():
            row = json.loads(path.read_text())
            if row.get("status") == "success":
                successful.append(row)
    best_block = min(
        successful,
        key=lambda row: row["peak_allocated_mib"],
    )["checkpoint_block_size"] if successful else 1

    for block in (0, best_block):
        output = checkpoint_output(root, 2048, int(block))
        run_child(
            [
                sys.executable,
                "student_scripts/a2k/checkpoint_benchmark.py",
                "--context-length",
                "2048",
                "--checkpoint-block-size",
                str(block),
                "--output",
                str(output),
                *common_flags(args),
            ],
            log_path=root / "logs" / "checkpoint" / f"s2048_ckpt{block}.log",
            expected_output=output,
            rerun=args.rerun,
        )


def run_compile(root: Path, args: argparse.Namespace) -> None:
    for implementation in ("torch_eager", "torch_compiled"):
        for mode in ("forward", "forward_backward", "train_step"):
            output = (
                root
                / "raw"
                / "compile_model"
                / f"{implementation}_{mode}.json"
            )
            run_child(
                [
                    sys.executable,
                    "student_scripts/a2k/compile_model_benchmark.py",
                    "--implementation",
                    implementation,
                    "--mode",
                    mode,
                    "--output",
                    str(output),
                    *common_flags(args),
                ],
                log_path=(
                    root
                    / "logs"
                    / "compile_model"
                    / f"{implementation}_{mode}.log"
                ),
                expected_output=output,
                rerun=args.rerun,
            )


def attention_output(
    root: Path,
    implementation: str,
    seq_len: int,
    head_dim: int,
    phase: str,
) -> Path:
    return (
        root
        / "raw"
        / "attention"
        / f"{implementation}_s{seq_len}_d{head_dim}_{phase}.json"
    )


def run_attention_one(
    root: Path,
    args: argparse.Namespace,
    implementation: str,
    seq_len: int,
    head_dim: int,
    phase: str,
) -> None:
    output = attention_output(root, implementation, seq_len, head_dim, phase)
    run_child(
        [
            sys.executable,
            "student_scripts/a2k/attention_benchmark.py",
            "--implementation",
            implementation,
            "--seq-len",
            str(seq_len),
            "--head-dim",
            str(head_dim),
            "--phase",
            phase,
            "--output",
            str(output),
            *common_flags(args),
        ],
        log_path=(
            root
            / "logs"
            / "attention"
            / f"{implementation}_s{seq_len}_d{head_dim}_{phase}.log"
        ),
        expected_output=output,
        rerun=args.rerun,
    )


def run_attention(root: Path, args: argparse.Namespace) -> None:
    for seq_len in (512, 2048, 8192):
        for head_dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                for implementation in (
                    "torch_eager",
                    "torch_compiled",
                    "flash_triton",
                ):
                    run_attention_one(
                        root,
                        args,
                        implementation,
                        seq_len,
                        head_dim,
                        phase,
                    )
    for head_dim in (64, 128):
        for phase in ("forward", "backward", "forward_backward"):
            for implementation in ("torch_eager", "flash_triton"):
                run_attention_one(
                    root,
                    args,
                    implementation,
                    16384,
                    head_dim,
                    phase,
                )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root",
        default="local_results/a2k",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        choices=(*CATEGORIES, "all"),
        default=["all"],
    )
    parser.add_argument("--rerun", action="store_true")
    parser.add_argument("--allow-nonstandard-gpu", action="store_true")
    parser.add_argument("--allow-low-free-memory", action="store_true")
    args = parser.parse_args()
    root = Path(args.root)
    selected = CATEGORIES if "all" in args.categories else args.categories
    runners = {
        "tests": run_tests,
        "correctness": run_correctness,
        "checkpoint": run_checkpoint,
        "compile": run_compile,
        "attention": run_attention,
    }
    for category in selected:
        print(f"\n===== {category} =====")
        runners[category](root, args)

    subprocess.run(
        [
            sys.executable,
            "student_scripts/a2k/summarize.py",
            "--root",
            str(root),
        ],
        check=True,
    )


if __name__ == "__main__":
    main()
