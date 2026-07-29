"""Run and combine the required 2x3 compute-profile matrix."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path

from profiling.common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--trace-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.trace_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    combined: list[dict] = []
    metadata: list[dict] = []
    for model_size in ("small", "medium"):
        for context_length in (256, 512, 1024):
            command = [
                sys.executable,
                "-m",
                "profiling.profile_runner",
                "--model-size",
                model_size,
                "--context-length",
                str(context_length),
                "--output-directory",
                str(args.trace_directory),
            ]
            subprocess.run(command, env=environment, check=True)
            run_id = f"{model_size}_ctx{context_length}"
            with (args.trace_directory / f"{run_id}_summary.csv").open(encoding="utf-8", newline="") as handle:
                combined.extend(csv.DictReader(handle))
            metadata.append(read_json(args.trace_directory / f"{run_id}_metadata.json"))
    write_csv(args.output, combined)
    write_json(
        args.metadata_output,
        {
            "primary_tool": "torch.profiler",
            "matrix": {
                "model_sizes": ["small", "medium"],
                "context_lengths": [256, 512, 1024],
                "mode": "train_step",
            },
            "runs": metadata,
            "raw_artifacts_policy": ("Full Chrome traces remain local and are not submitted."),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
