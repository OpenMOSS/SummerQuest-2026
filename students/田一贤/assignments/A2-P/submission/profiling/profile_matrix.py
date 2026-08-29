"""Run the required two-model by three-context profiling matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-root", type=Path, default=Path("results/profile_matrix"))
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    configs = [(model, context) for model in ("small", "medium") for context in (256, 512, 1024)]
    manifest = []
    for model, context in configs:
        tag = f"{model}_c{context}_{args.dtype}"
        output_dir = args.output_root / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            "-m",
            "profiling.compute_profile",
            "--model-size",
            model,
            "--context-length",
            str(context),
            "--batch-size",
            str(args.batch_size),
            "--dtype",
            args.dtype,
            "--warmup-steps",
            str(args.warmup_steps),
            "--seed",
            str(args.seed),
            "--trace-name",
            f"{tag}.trace.json",
            "--output-dir",
            str(output_dir),
        ]
        subprocess.run(command, check=True)
        trace_path = output_dir / f"{tag}.trace.json"
        summarize_command = [
            sys.executable,
            "-m",
            "profiling.trace_summarize",
            "--trace",
            str(trace_path),
            "--output-dir",
            str(output_dir),
            "--dtype",
            args.dtype,
            "--model-size",
            model,
            "--context-length",
            str(context),
            "--batch-size",
            str(args.batch_size),
            "--warmup-steps",
            str(args.warmup_steps),
            "--seed",
            str(args.seed),
        ]
        subprocess.run(summarize_command, check=True)
        metadata_path = output_dir / "run_metadata.json"
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        metadata.update(
            {
                "matrix_tag": tag,
                "matrix_model_size": model,
                "matrix_context_length": context,
                "matrix_dtype": args.dtype,
                "matrix_batch_size": args.batch_size,
                "matrix_seed": args.seed,
                "matrix_profiled_steps": 1,
                "matrix_command": " ".join(command),
                "summary_command": " ".join(summarize_command),
                "trace_file": str(Path(tag) / metadata.get("trace_file", f"{tag}.trace.json")),
            }
        )
        metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
        manifest.append(metadata)

    with (args.output_root / "matrix_manifest.json").open("w", encoding="utf-8") as f:
        stage_ranges = [
            "profile/warmup",
            "profile/measure",
            "forward",
            "attention",
            "attention/scores",
            "attention/softmax",
            "attention/value",
            "backward",
            "optimizer",
        ]
        json.dump(
            {
                "status": "pass",
                "tool": "torch.profiler Chrome trace",
                "evaluation_type": "self_supervised_proxy",
                "models": ["small", "medium"],
                "contexts": [256, 512, 1024],
                "dtype": args.dtype,
                "batch_size": args.batch_size,
                "warmup_train_step_steps": args.warmup_steps,
                "profiled_train_step_steps": 1,
                "seed": args.seed,
                "stage_ranges": stage_ranges,
                "configurations": manifest,
            },
            f,
            indent=2,
        )

    trace_fields = [
        "model_size", "context_length", "dtype", "name", "calls",
        "cpu_time_us", "cuda_time_us", "status", "trace_file",
    ]
    stage_fields = [
        "model_size", "context_length", "dtype", "stage", "cpu_calls",
        "cpu_total_us", "cuda_annotation_calls", "cuda_annotation_total_us",
        "cuda_kernel_calls", "cuda_kernel_total_us", "status", "trace_file",
    ]
    trace_rows, stage_rows = [], []
    for metadata in manifest:
        tag = metadata["matrix_tag"]
        output_dir = args.output_root / tag
        trace_file = metadata["trace_file"]
        with (output_dir / "trace_summary.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                trace_rows.append({
                    "model_size": metadata["matrix_model_size"],
                    "context_length": metadata["matrix_context_length"],
                    "dtype": metadata["matrix_dtype"],
                    **row,
                    "trace_file": trace_file,
                })
        with (output_dir / "stage_summary.csv").open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stage_rows.append({
                    "model_size": metadata["matrix_model_size"],
                    "context_length": metadata["matrix_context_length"],
                    "dtype": metadata["matrix_dtype"],
                    **row,
                    "trace_file": trace_file,
                })
    with (args.output_root / "trace_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=trace_fields)
        writer.writeheader()
        writer.writerows(trace_rows)
    with (args.output_root / "stage_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=stage_fields)
        writer.writeheader()
        writer.writerows(stage_rows)


if __name__ == "__main__":
    main()
