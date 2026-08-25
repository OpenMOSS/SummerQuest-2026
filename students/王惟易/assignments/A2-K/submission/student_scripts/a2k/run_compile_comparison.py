from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


ATTENTION_CONFIGS = ((512, 64), (2048, 128), (8192, 128))
ATTENTION_PHASES = ("forward", "backward", "forward_backward")
TRANSFORMER_PHASES = ("forward", "forward_backward", "train_step")
IMPLEMENTATIONS = ("eager", "compiled")
CSV_FIELDS = (
    "config_id",
    "workload",
    "implementation",
    "model_size",
    "batch_size",
    "sequence_length",
    "head_dim",
    "dtype",
    "is_causal",
    "phase",
    "warmup_ms",
    "rep_ms",
    "measurement_count",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "speedup_vs_eager",
    "cold_start_total_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "within_24gib",
    "status",
    "failure_stage",
    "error_type",
    "source_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed eager/compiled attention and Transformer comparison.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--minimum-free-mib", type=float, default=22 * 1024)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def cases() -> Iterator[tuple[str, str, int | None, int | None, str]]:
    for sequence_length, head_dim in ATTENTION_CONFIGS:
        for phase in ATTENTION_PHASES:
            for implementation in IMPLEMENTATIONS:
                yield "attention", implementation, sequence_length, head_dim, phase

    for phase in TRANSFORMER_PHASES:
        for implementation in IMPLEMENTATIONS:
            yield "transformer_small", implementation, None, None, phase


def result_path(
    output_dir: Path,
    workload: str,
    implementation: str,
    sequence_length: int | None,
    head_dim: int | None,
    phase: str,
) -> Path:
    if workload == "attention":
        return output_dir / f"attention_{implementation}_b1_t{sequence_length}_d{head_dim}_{phase}_bf16.json"
    return output_dir / f"transformer_small_{implementation}_b1_t512_{phase}_bf16.json"


def load_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"ok", "oom"}:
        return None
    return result


def run_case(
    args: argparse.Namespace,
    workload: str,
    implementation: str,
    sequence_length: int | None,
    head_dim: int | None,
    phase: str,
) -> tuple[dict[str, Any], Path]:
    output = result_path(
        args.output_dir,
        workload,
        implementation,
        sequence_length,
        head_dim,
        phase,
    )
    if args.resume and (completed := load_completed(output)) is not None:
        print(f"Reusing {output}", flush=True)
        return completed, output

    if workload == "attention":
        command = [
            sys.executable,
            "student_scripts/a2k/attention_benchmark.py",
            "--sequence-length",
            str(sequence_length),
            "--head-dim",
            str(head_dim),
            "--implementation",
            implementation,
            "--phase",
            phase,
            "--warmup-ms",
            "100",
            "--rep-ms",
            "300",
            "--seed",
            str(args.seed),
            "--minimum-free-mib",
            str(args.minimum_free_mib),
            "--output",
            output.as_posix(),
        ]
    else:
        command = [
            sys.executable,
            "student_scripts/a2k/transformer_compile_benchmark.py",
            "--implementation",
            implementation,
            "--phase",
            phase,
            "--warmup-ms",
            "100",
            "--rep-ms",
            "300",
            "--seed",
            str(args.seed),
            "--learning-rate",
            str(args.learning_rate),
            "--minimum-free-mib",
            str(args.minimum_free_mib),
            "--output",
            output.as_posix(),
        ]

    print(f"Running {workload}: {implementation}, phase={phase}, T={sequence_length}, D={head_dim}", flush=True)
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8")), output


def to_csv_row(
    workload: str,
    result: dict[str, Any],
    source: Path,
) -> dict[str, Any]:
    config = result["config"]
    timing = result.get("timing", {})
    memory = result.get("memory", {})
    allocator = result["allocator"]
    cold_start = result.get("cold_start", {})
    sequence_length = config.get("sequence_length", config.get("context_length"))
    head_dim = config.get("head_dim", "")
    return {
        "config_id": f"{workload}-{config['implementation']}-b1-t{sequence_length}-d{head_dim or 'na'}-{config['phase']}-bf16",
        "workload": workload,
        "implementation": config["implementation"],
        "model_size": config.get("model_size", ""),
        "batch_size": config["batch_size"],
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "dtype": config["dtype"],
        "is_causal": str(config.get("is_causal", True)).lower(),
        "phase": config["phase"],
        "warmup_ms": config["warmup_ms"],
        "rep_ms": config["rep_ms"],
        "measurement_count": timing.get("measurement_count", ""),
        "p20_ms": timing.get("p20_ms", ""),
        "p50_ms": timing.get("p50_ms", ""),
        "p80_ms": timing.get("p80_ms", ""),
        "speedup_vs_eager": "",
        "cold_start_total_ms": cold_start.get("total_ms", ""),
        "peak_allocated_mib": memory.get("peak_allocated_mib", ""),
        "peak_reserved_mib": memory.get("peak_reserved_mib", ""),
        "allocator_limit_mib": allocator["allocator_limit_mib"],
        "allocator_fraction": allocator["allocator_fraction"],
        "within_24gib": str(memory.get("within_24gib", False)).lower(),
        "status": result["status"],
        "failure_stage": result.get("failure_stage") or "",
        "error_type": result.get("error_type") or "",
        "source_json": source.name,
    }


def speedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row["workload"],
        row["model_size"],
        row["batch_size"],
        row["sequence_length"],
        row["head_dim"],
        row["dtype"],
        row["is_causal"],
        row["phase"],
    )


def add_speedups(rows: list[dict[str, Any]]) -> None:
    eager_p50 = {
        speedup_key(row): float(row["p50_ms"])
        for row in rows
        if row["implementation"] == "eager" and row["status"] == "ok"
    }
    for row in rows:
        baseline = eager_p50.get(speedup_key(row))
        if baseline is not None and row["status"] == "ok":
            row["speedup_vs_eager"] = baseline / float(row["p50_ms"])


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []

    for workload, implementation, sequence_length, head_dim, phase in cases():
        result, source = run_case(
            args,
            workload,
            implementation,
            sequence_length,
            head_dim,
            phase,
        )
        rows.append(to_csv_row(workload, result, source))

    add_speedups(rows)
    output = args.output_dir / "compile_comparison.csv"
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
