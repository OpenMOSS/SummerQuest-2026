from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from pathlib import Path

import torch


def _run(module: str, *args: object) -> None:
    command = [sys.executable, "-m", module, *(str(item) for item in args)]
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results", type=Path, default=Path("results"))
    parser.add_argument("--formal", action="store_true")
    parser.add_argument("--include-boundary", action="store_true")
    parser.add_argument("--skip-compiled-boundary", action="store_true")
    args = parser.parse_args()
    args.results.mkdir(parents=True, exist_ok=True)
    metadata = {
        "student": "田一贤",
        "status": "formal_run_started" if args.formal else "formal_run_not_requested",
        "measurement_collected": False,
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "allocator_limit_mib": 23552,
        "allocator_guard": "each benchmark subprocess applies torch.cuda.set_per_process_memory_fraction before first CUDA tensor allocation",
        "execution_order": ["correctness", "checkpoint", "attention", "compile", "flash", "memory_evidence"],
    }
    metadata_path = args.results / "run_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    if not args.formal:
        print(json.dumps(metadata, indent=2))
        return
    if not torch.cuda.is_available():
        raise RuntimeError("--formal requires CUDA")

    _run(
        "student_scripts.a2k.correctness",
        "--device",
        "cuda",
        "--output",
        args.results / "correctness.json",
    )
    checkpoint_args = ["--output", args.results / "checkpointing.csv", "--include-2048"]
    _run("student_scripts.a2k.checkpoint_benchmark", *checkpoint_args)
    _run(
        "student_scripts.a2k.attention_benchmark",
        "--output",
        args.results / "attention_baseline.csv",
    )
    _run(
        "student_scripts.a2k.compile_benchmark",
        "--output",
        args.results / "compile_comparison.csv",
    )
    flash_args = ["--output", args.results / "flash_benchmark.csv"]
    if args.include_boundary:
        flash_args.append("--include-boundary")
    if args.skip_compiled_boundary:
        flash_args.append("--skip-compiled-boundary")
    _run("student_scripts.a2k.flash_benchmark", *flash_args)
    _run(
        "student_scripts.a2k.memory_evidence",
        "--results",
        args.results,
        "--output",
        args.results / "memory_evidence.json",
    )
    metadata.update(status="formal_run_complete", measurement_collected=True)
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
