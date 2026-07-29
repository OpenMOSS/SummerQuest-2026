"""Serially run the fixed checkpoint matrix using a fresh process per row."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
from pathlib import Path

from student_scripts.a2k.common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    return parser.parse_args()


def run_one(context: int, block: str, destination: Path) -> dict:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.checkpoint_benchmark",
        "--context-length",
        str(context),
        "--checkpoint-block-size",
        block,
        "--output",
        str(destination),
    ]
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    subprocess.run(command, env=environment, check=True)
    return read_json(destination)


def main() -> int:
    args = parse_args()
    rows: list[dict] = []
    metadata: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="a2k-checkpoint-") as temporary:
        scratch = Path(temporary)
        for block in ("none", "1", "2", "4", "8"):
            payload = run_one(1024, block, scratch / f"1024-{block}.json")
            rows.append(payload["row"])
            metadata.append(
                {
                    "config_id": payload["row"]["config_id"],
                    "allocator": payload["allocator"],
                    "gpu": payload["gpu"],
                    "command": payload["command"],
                }
            )

        successful = [row for row in rows if row["status"] == "ok" and row["checkpoint_block_size"] != "none"]
        if not successful:
            raise RuntimeError("no checkpointed context-1024 configuration succeeded")
        best = min(successful, key=lambda row: float(row["peak_allocated_mib"]))
        best_block = str(best["checkpoint_block_size"])
        for block in ("none", best_block):
            payload = run_one(2048, block, scratch / f"2048-{block}.json")
            rows.append(payload["row"])
            metadata.append(
                {
                    "config_id": payload["row"]["config_id"],
                    "allocator": payload["allocator"],
                    "gpu": payload["gpu"],
                    "command": payload["command"],
                }
            )

    write_csv(args.output, rows)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "experiment": "activation_checkpointing",
                "process_isolation": "one fresh Python process per row",
                "runs": metadata,
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
