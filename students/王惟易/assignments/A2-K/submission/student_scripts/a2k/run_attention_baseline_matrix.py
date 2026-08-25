from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


SEQUENCE_LENGTHS = (512, 2048, 8192)
HEAD_DIMS = (64, 128)
PHASES = ("forward", "backward", "forward_backward")
CSV_FIELDS = (
    "config_id",
    "implementation",
    "batch_size",
    "sequence_length",
    "head_dim",
    "dtype",
    "is_causal",
    "phase",
    "warmup_ms",
    "rep_ms",
    "quantiles",
    "measurement_count",
    "p20_ms",
    "p50_ms",
    "p80_ms",
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
    parser = argparse.ArgumentParser(description="Run the fixed eager-attention baseline matrix.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--minimum-free-mib", type=float, default=22 * 1024)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def result_path(output_dir: Path, sequence_length: int, head_dim: int, phase: str) -> Path:
    return output_dir / f"eager_b1_t{sequence_length}_d{head_dim}_{phase}_bf16.json"


def load_completed(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    result = json.loads(path.read_text(encoding="utf-8"))
    if result.get("status") not in {"ok", "oom"}:
        return None
    return result


def run_case(
    args: argparse.Namespace,
    sequence_length: int,
    head_dim: int,
    phase: str,
) -> dict[str, Any]:
    output = result_path(args.output_dir, sequence_length, head_dim, phase)
    if args.resume and (completed := load_completed(output)) is not None:
        print(f"Reusing {output}", flush=True)
        return completed

    command = [
        sys.executable,
        "student_scripts/a2k/attention_benchmark.py",
        "--sequence-length",
        str(sequence_length),
        "--head-dim",
        str(head_dim),
        "--implementation",
        "eager",
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
        f"Running eager attention T={sequence_length}, D={head_dim}, phase={phase}",
        flush=True,
    )
    subprocess.run(command, check=True)
    return json.loads(output.read_text(encoding="utf-8"))


def to_csv_row(result: dict[str, Any], source: Path) -> dict[str, Any]:
    config = result["config"]
    timing = result.get("timing", {})
    memory = result["memory"]
    allocator = result["allocator"]
    return {
        "config_id": (f"eager-b1-t{config['sequence_length']}-d{config['head_dim']}-{config['phase']}-bf16"),
        "implementation": config["implementation"],
        "batch_size": config["batch_size"],
        "sequence_length": config["sequence_length"],
        "head_dim": config["head_dim"],
        "dtype": config["dtype"],
        "is_causal": str(config["is_causal"]).lower(),
        "phase": config["phase"],
        "warmup_ms": config["warmup_ms"],
        "rep_ms": config["rep_ms"],
        "quantiles": json.dumps(config["quantiles"], separators=(",", ":")),
        "measurement_count": timing.get("measurement_count", ""),
        "p20_ms": timing.get("p20_ms", ""),
        "p50_ms": timing.get("p50_ms", ""),
        "p80_ms": timing.get("p80_ms", ""),
        "peak_allocated_mib": memory["peak_allocated_mib"],
        "peak_reserved_mib": memory["peak_reserved_mib"],
        "allocator_limit_mib": allocator["allocator_limit_mib"],
        "allocator_fraction": allocator["allocator_fraction"],
        "within_24gib": str(memory["within_24gib"]).lower(),
        "status": result["status"],
        "failure_stage": result.get("failure_stage") or "",
        "error_type": result.get("error_type") or "",
        "source_json": source.name,
    }


def write_csv(output: Path, results: list[tuple[dict[str, Any], Path]]) -> None:
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        for result, source in results:
            writer.writerow(to_csv_row(result, source))


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[dict[str, Any], Path]] = []
    for sequence_length in SEQUENCE_LENGTHS:
        for head_dim in HEAD_DIMS:
            for phase in PHASES:
                source = result_path(args.output_dir, sequence_length, head_dim, phase)
                results.append(
                    (
                        run_case(args, sequence_length, head_dim, phase),
                        source,
                    )
                )

    csv_output = args.output_dir / "attention_baseline.csv"
    write_csv(csv_output, results)
    print(f"Wrote {len(results)} rows to {csv_output}")


if __name__ == "__main__":
    main()
