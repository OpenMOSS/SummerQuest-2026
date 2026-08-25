from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any


CORE_SEQUENCE_LENGTHS = (512, 2048, 8192)
BOUNDARY_SEQUENCE_LENGTH = 16384
HEAD_DIMS = (64, 128)
PHASES = ("forward", "backward", "forward_backward")
CORE_IMPLEMENTATIONS = ("eager", "compiled", "triton")
BOUNDARY_IMPLEMENTATIONS = ("eager", "triton")
CSV_FIELDS = (
    "config_id",
    "implementation",
    "batch_size",
    "sequence_length",
    "head_dim",
    "dtype",
    "is_causal",
    "phase",
    "is_boundary",
    "warmup_ms",
    "rep_ms",
    "quantiles",
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
    "query_tile_size",
    "key_tile_size",
    "num_warps",
    "num_stages",
    "status",
    "failure_stage",
    "error_type",
    "source_json",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the fixed A2-K FlashAttention performance matrix serially.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-mib", type=float, default=22 * 1024)
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse completed ok/oom JSON cases in the output directory.",
    )
    return parser.parse_args()


def matrix_cases() -> Iterator[tuple[str, int, int, str, bool]]:
    for sequence_length in CORE_SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for phase in PHASES:
                for implementation in CORE_IMPLEMENTATIONS:
                    yield implementation, sequence_length, head_dim, phase, False

    for head_dim in HEAD_DIMS:
        for phase in PHASES:
            for implementation in BOUNDARY_IMPLEMENTATIONS:
                yield implementation, BOUNDARY_SEQUENCE_LENGTH, head_dim, phase, True


def result_path(
    output_dir: Path,
    implementation: str,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> Path:
    return output_dir / f"{implementation}_b1_t{sequence_length}_d{head_dim}_{phase}_bf16.json"


def load_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"ok", "oom"}:
        return None
    return result


def run_case(
    args: argparse.Namespace,
    implementation: str,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> tuple[dict[str, Any], Path]:
    output = result_path(
        args.output_dir,
        implementation,
        sequence_length,
        head_dim,
        phase,
    )
    if args.resume and (completed := load_completed(output)) is not None:
        print(f"Reusing {output}", flush=True)
        return completed, output

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
    print(
        f"Running {implementation}: T={sequence_length}, D={head_dim}, phase={phase}",
        flush=True,
    )
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8")), output


def to_csv_row(
    result: dict[str, Any],
    source: Path,
    is_boundary: bool,
) -> dict[str, Any]:
    config = result["config"]
    timing = result.get("timing", {})
    memory = result.get("memory", {})
    allocator = result["allocator"]
    triton_launch = config.get("triton_launch") or {}
    cold_start = result.get("cold_start", {})
    return {
        "config_id": (
            f"{config['implementation']}-b1-t{config['sequence_length']}-"
            f"d{config['head_dim']}-{config['phase']}-bf16"
        ),
        "implementation": config["implementation"],
        "batch_size": config["batch_size"],
        "sequence_length": config["sequence_length"],
        "head_dim": config["head_dim"],
        "dtype": config["dtype"],
        "is_causal": str(config["is_causal"]).lower(),
        "phase": config["phase"],
        "is_boundary": str(is_boundary).lower(),
        "warmup_ms": config["warmup_ms"],
        "rep_ms": config["rep_ms"],
        "quantiles": json.dumps(config["quantiles"], separators=(",", ":")),
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
        "query_tile_size": triton_launch.get("query_tile_size", ""),
        "key_tile_size": triton_launch.get("key_tile_size", ""),
        "num_warps": triton_launch.get("num_warps", ""),
        "num_stages": triton_launch.get("num_stages", ""),
        "status": result["status"],
        "failure_stage": result.get("failure_stage") or "",
        "error_type": result.get("error_type") or "",
        "source_json": source.name,
    }


def speedup_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
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
        if baseline is None or row["status"] != "ok":
            continue
        row["speedup_vs_eager"] = baseline / float(row["p50_ms"])


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for implementation, sequence_length, head_dim, phase, is_boundary in matrix_cases():
        result, source = run_case(
            args,
            implementation,
            sequence_length,
            head_dim,
            phase,
        )
        rows.append(to_csv_row(result, source, is_boundary))

    add_speedups(rows)
    output = args.output_dir / "flash_benchmark.csv"
    write_csv(output, rows)
    print(f"Wrote {len(rows)} rows to {output}")


if __name__ == "__main__":
    main()
