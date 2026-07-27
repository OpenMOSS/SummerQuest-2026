#!/usr/bin/env python3
"""Run the A2-K checkpoint matrix with one fresh process per configuration."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import tempfile
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("results/checkpointing.csv"))
    parser.add_argument("--model-size", default="medium")
    parser.add_argument("--context-lengths", default="1024,2048")
    parser.add_argument("--groups", default="0,1,2,4,8")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--dtype", default="bf16")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    rows = []
    for context_length in args.context_lengths.split(","):
        for group_size in args.groups.split(","):
            with tempfile.NamedTemporaryFile(suffix=".json") as temporary:
                command = [
                    sys.executable,
                    str(root / "scripts" / "checkpoint_benchmark.py"),
                    "--model-size",
                    args.model_size,
                    "--context-length",
                    context_length,
                    "--batch-size",
                    str(args.batch_size),
                    "--dtype",
                    args.dtype,
                    "--group-size",
                    group_size,
                    "--warmup",
                    str(args.warmup),
                    "--steps",
                    str(args.steps),
                    "--output",
                    temporary.name,
                ]
                try:
                    subprocess.run(command, cwd=root, check=True, capture_output=True, text=True)
                    rows.append(json.loads(Path(temporary.name).read_text()))
                except subprocess.CalledProcessError as exc:
                    rows.append(
                        {
                            "context_length": int(context_length),
                            "checkpoint_block_size": int(group_size),
                            "status": "error",
                            "error": (exc.stderr or exc.stdout or str(exc)).strip(),
                        }
                    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
