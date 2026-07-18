#!/usr/bin/env python3
"""Collect submission-safe logs, plots, and samples into one archive."""

from __future__ import annotations

import shutil
import subprocess
import sys
import tarfile
from pathlib import Path


def copy_if_exists(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def plot(logs: list[Path], labels: list[str], output: Path) -> None:
    existing = [(log, label) for log, label in zip(logs, labels) if log.exists()]
    if not existing:
        return
    output.parent.mkdir(parents=True, exist_ok=True)
    command = [sys.executable, "scripts/plot_logs.py"]
    for log, label in existing:
        command.extend(["--log", str(log), "--label", label])
    command.extend(["--output", str(output)])
    subprocess.run(command, check=True)


def main() -> None:
    export = Path("public_export")
    if export.exists():
        shutil.rmtree(export)
    (export / "logs").mkdir(parents=True)
    (export / "assets").mkdir(parents=True)

    for source in Path("logs/tokenizers").glob("*.json"):
        copy_if_exists(source, export / "logs/tokenizers" / source.name)

    for source in Path("runs").rglob("*"):
        if not source.is_file() or source.name not in {"train.jsonl", "summary.json"}:
            continue
        relative = source.relative_to("runs")
        if any(part in {"smoke", "full_model_probe"} for part in relative.parts):
            continue
        copy_if_exists(source, export / "logs/runs" / relative)

    generation_dir = Path("runs/generation")
    if generation_dir.exists():
        for source in generation_dir.glob("*.txt"):
            copy_if_exists(source, export / "samples" / source.name)

    plot(
        [Path("runs/tinystories_baseline/train.jsonl")],
        ["TinyStories baseline"],
        export / "assets/tinystories_loss.png",
    )
    plot(
        [
            Path("runs/lr_sweep/lr_1e-4/train.jsonl"),
            Path("runs/lr_sweep/lr_3e-4/train.jsonl"),
            Path("runs/lr_sweep/lr_1e-3/train.jsonl"),
            Path("runs/lr_sweep/lr_divergent_3e-2/train.jsonl"),
            Path("runs/lr_sweep/lr_divergent_3e-1/train.jsonl"),
        ],
        ["1e-4", "3e-4", "1e-3", "3e-2", "3e-1"],
        export / "assets/lr_sweep.png",
    )
    plot(
        [
            Path("runs/ablations/control/train.jsonl"),
            Path("runs/ablations/no_rmsnorm/train.jsonl"),
            Path("runs/ablations/post_norm/train.jsonl"),
            Path("runs/ablations/no_rope/train.jsonl"),
            Path("runs/ablations/silu/train.jsonl"),
        ],
        ["control", "no RMSNorm", "post-norm", "NoPE", "SiLU"],
        export / "assets/ablations.png",
    )
    plot(
        [Path("runs/owt_baseline/train.jsonl")],
        ["OWT baseline"],
        export / "assets/owt_loss.png",
    )

    with tarfile.open("a1-public-results.tar.gz", "w:gz") as archive:
        archive.add(export, arcname="public_export")
    print("created a1-public-results.tar.gz")


if __name__ == "__main__":
    main()
