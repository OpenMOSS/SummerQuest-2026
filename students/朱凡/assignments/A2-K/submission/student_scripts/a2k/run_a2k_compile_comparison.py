#!/usr/bin/env python3
"""Produce the A2-K attention and full-model compile comparison table."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def run_json(command: list[str], root: Path) -> object:
    completed = subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    return json.loads(lines[-1])


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/compile_comparison.csv"))
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repetitions", type=int, default=300)
    parser.add_argument("--allocator-limit-mib", type=int, default=23552)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
        command = [
            sys.executable,
            str(root / "scripts" / "attention_benchmark.py"),
            "--implementations",
            "eager,compile",
            "--sequence-lengths",
            "512,2048,8192",
            "--dimensions",
            "64,128",
            "--dtypes",
            "bfloat16",
            "--batch-size",
            "1",
            "--causal",
            "--warmup",
            str(args.warmup),
            "--repetitions",
            str(args.repetitions),
            "--output",
            temporary.name,
            "--allocator-limit-mib",
            str(args.allocator_limit_mib),
        ]
        try:
            subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
            rows.extend(json.loads(Path(temporary.name).read_text()))
        except subprocess.CalledProcessError as exc:
            rows.append(
                {
                    "experiment": "attention",
                    "status": "error",
                    "error": (exc.stderr or exc.stdout or str(exc)).strip(),
                    "command": " ".join(command),
                }
            )

    for mode in ("forward", "forward_backward", "train_step"):
        for compiled in (False, True):
            with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
                command = [
                    sys.executable,
                    str(root / "scripts" / "benchmark.py"),
                    "--model-size",
                    "small",
                    "--context-length",
                    "512",
                    "--batch-size",
                    "1",
                    "--dtype",
                    "bf16",
                    "--mode",
                    mode,
                    "--warmup",
                    "5",
                    "--steps",
                    "10",
                    "--json",
                    "--output",
                    temporary.name,
                    "--allocator-limit-mib",
                    str(args.allocator_limit_mib),
                ]
                if compiled:
                    command.append("--compile")
                try:
                    row = run_json(command, root)
                    row["experiment"] = "full_model"
                    row["compiled"] = compiled
                except subprocess.CalledProcessError as exc:
                    row = {
                        "experiment": "full_model",
                        "compiled": compiled,
                        "mode": mode,
                        "status": "error",
                        "error": (exc.stderr or exc.stdout or str(exc)).strip(),
                    }
                rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
