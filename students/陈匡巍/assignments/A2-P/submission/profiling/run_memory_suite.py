"""Run required XL memory rows and the prescribed fallback when needed."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

from profiling.common import read_json, write_csv, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--local-directory", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.local_directory.mkdir(parents=True, exist_ok=True)
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    rows: list[dict] = []
    runs: list[dict] = []

    def run(model_size: str, context_length: int, mode: str) -> dict:
        run_id = f"{model_size}_ctx{context_length}_{mode}"
        output = args.local_directory / f"{run_id}.json"
        snapshot = args.local_directory / f"{run_id}.pickle"
        command = [
            sys.executable,
            "-m",
            "profiling.memory_snapshot",
            "--model-size",
            model_size,
            "--context-length",
            str(context_length),
            "--mode",
            mode,
            "--output",
            str(output),
            "--snapshot",
            str(snapshot),
        ]
        subprocess.run(command, env=environment, check=True)
        return read_json(output)

    for context_length in (128, 2048):
        for mode in ("forward", "train_step"):
            payload = run("xl", context_length, mode)
            rows.append(payload["row"])
            runs.append(payload)

    xl_2048_train = next(row for row in rows if row["model_size"] == "xl" and row["context_length"] == 2048 and row["mode"] == "train_step")
    if xl_2048_train["status"] != "ok":
        payload = run("xl", 1024, "train_step")
        rows.append(payload["row"])
        runs.append(payload)
        if payload["row"]["status"] != "ok":
            payload = run("large", 2048, "train_step")
            rows.append(payload["row"])
            runs.append(payload)
            if payload["row"]["status"] != "ok":
                # This does not replace either prescribed fallback.  It gives
                # a successful full-step timeline for stage/residual analysis
                # after all required 23 GiB boundary rows have been retained.
                payload = run("large", 128, "train_step")
                rows.append(payload["row"])
                runs.append(payload)

    write_csv(args.output, rows)
    write_json(
        args.metadata_output,
        {
            "experiment": "memory_profiling",
            "dtype": "fp32",
            "warmup_steps": 1,
            "measurement_steps": 1,
            "history": "torch.cuda.memory._record_memory_history",
            "runs": runs,
            "supplemental_diagnostic": ("Large/context-128 train_step is added only if both prescribed training fallbacks OOM; it never substitutes for them."),
            "residual_stream_formula": "batch * context * d_model * bytes_per_element",
            "raw_artifacts_policy": ("Pickled snapshots remain local and are not submitted."),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
