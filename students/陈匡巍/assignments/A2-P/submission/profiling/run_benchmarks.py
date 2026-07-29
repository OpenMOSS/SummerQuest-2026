"""Run the required small-model benchmark rows in isolated processes."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from profiling.common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    configurations = [
        ("forward", 5),
        ("forward_backward", 5),
        ("train_step", 5),
        ("train_step", 0),
    ]
    rows: list[dict] = []
    runs: list[dict] = []
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    with tempfile.TemporaryDirectory(prefix="a2p-benchmark-") as temporary:
        for index, (mode, warmup) in enumerate(configurations):
            destination = Path(temporary) / f"{index}.json"
            command = [
                sys.executable,
                "-m",
                "profiling.benchmark",
                "--model-size",
                "small",
                "--batch-size",
                "4",
                "--context-length",
                "512",
                "--mode",
                mode,
                "--warmup",
                str(warmup),
                "--steps",
                "10",
                "--dtype",
                "fp32",
                "--output",
                str(destination),
            ]
            subprocess.run(command, env=environment, check=True)
            payload = read_json(destination)
            rows.append(payload["row"])
            runs.append(
                {
                    "command": payload["command"],
                    "allocator": payload["allocator"],
                    "gpu": payload["gpu"],
                    "software": payload["software"],
                    "timer": payload["timer"],
                }
            )
    write_csv(args.output, rows)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "experiment": "end_to_end_benchmark",
                "process_isolation": "one fresh Python process per row",
                "runs": runs,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
