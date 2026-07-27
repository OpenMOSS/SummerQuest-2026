from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
from pathlib import Path

ALLOCATOR_LIMIT_MIB = 23 * 1024
HARD_LIMIT_MIB = 24 * 1024
REPRESENTATIVE_COMPILE_SHAPES = {(512, 64), (2048, 128), (8192, 128)}


def speedup(reference_ms: str | None, candidate_ms: str | None) -> float | None:
    try:
        reference = float(reference_ms)
        candidate = float(candidate_ms)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(reference) or not math.isfinite(candidate) or candidate <= 0:
        return None
    return reference / candidate


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"cannot write an empty result file: {path}")
    fields = fieldnames or list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def matching_key(row: dict[str, str]) -> tuple[str, ...]:
    return (
        row["batch_size"],
        row["sequence_length"],
        row["head_dimension"],
        row["dtype"],
        row["causal"],
        row["phase"],
        row["warmup_ms"],
        row["rep_ms"],
    )


def add_speedups(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    eager = {matching_key(row): row for row in rows if row["implementation"] == "eager" and row["status"] == "ok"}
    output = []
    for source in rows:
        row: dict[str, object] = dict(source)
        reference = eager.get(matching_key(source))
        value = None
        if reference is not None and source["status"] == "ok":
            value = speedup(reference["latency_ms_p50"], source["latency_ms_p50"])
        row["speedup_vs_eager_p50"] = "" if value is None else value
        output.append(row)
    return output


def peak(rows: list[dict[str, object]], field: str) -> float:
    values = []
    for row in rows:
        try:
            values.append(float(row.get(field) or 0.0))
        except (TypeError, ValueError):
            continue
    return max(values, default=0.0)


def load_environment(raw_root: Path) -> dict:
    for name in ("attention_metadata.json", "checkpointing_metadata.json", "model_compile_metadata.json"):
        path = raw_root / name
        if path.is_file():
            return json.loads(path.read_text())["environment"]
    raise FileNotFoundError("no A2-K environment metadata found")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight public A2-K result files.")
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--public-root", type=Path, required=True)
    parser.add_argument("--starter-commit", default="ca8bc81a59b70516f7ebb2da4808daade877c736")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.public_root.mkdir(parents=True, exist_ok=True)
    attention_rows = read_csv(args.raw_root / "flash_benchmark.csv")
    flash_rows = add_speedups(attention_rows)
    write_csv(args.public_root / "flash_benchmark.csv", flash_rows)

    baseline = [row for row in attention_rows if row["implementation"] == "eager" and int(row["sequence_length"]) <= 8192]
    write_csv(args.public_root / "attention_baseline.csv", baseline)

    compile_rows: list[dict[str, object]] = [
        {"scope": "attention", **row}
        for row in attention_rows
        if row["implementation"] in {"eager", "compiled"} and (int(row["sequence_length"]), int(row["head_dimension"])) in REPRESENTATIVE_COMPILE_SHAPES
    ]
    model_compile = args.raw_root / "model_compile.csv"
    if model_compile.is_file():
        compile_rows.extend(read_csv(model_compile))
    compile_fields = sorted({key for row in compile_rows for key in row})
    write_csv(args.public_root / "compile_comparison.csv", compile_rows, compile_fields)

    checkpoint_rows = read_csv(args.raw_root / "checkpointing.csv")
    shutil.copy2(args.raw_root / "checkpointing.csv", args.public_root / "checkpointing.csv")
    shutil.copy2(args.raw_root / "correctness.json", args.public_root / "correctness.json")
    unit_tests = args.raw_root / "unit_tests.txt"
    if unit_tests.is_file():
        shutil.copy2(unit_tests, args.public_root / "unit_tests.txt")

    all_rows: list[dict[str, object]] = [*flash_rows, *checkpoint_rows, *compile_rows]
    environment = load_environment(args.raw_root)
    allocator_fraction = float(environment["allocator_fraction"])
    peak_allocated = peak(all_rows, "peak_allocated_mib")
    peak_reserved = peak(all_rows, "peak_reserved_mib")
    evidence = {
        "allocator": {
            "allocator_fraction": allocator_fraction,
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": peak_allocated,
        "pytorch_peak_reserved_mib": peak_reserved,
        "within_24gib": peak_reserved <= ALLOCATOR_LIMIT_MIB and peak_allocated <= HARD_LIMIT_MIB,
        "observed_gpu_total_memory_mib": environment.get("gpu_total_memory_mib"),
        "formal_24gb_capacity_confirmed": 22000 <= float(environment.get("gpu_total_memory_mib", 0)) <= 25000,
    }
    (args.public_root / "memory_evidence.json").write_text(json.dumps(evidence, indent=2) + "\n")
    metadata = {
        "starter_commit": args.starter_commit,
        "seed": 2026,
        "environment": environment,
        "measurement": {
            "attention": {"timer": "triton.testing.do_bench", "warmup_ms": 100, "rep_ms": 300},
            "checkpointing": {"warmup_steps": 3, "measurement_steps": 5},
            "model_compile": {"timer": "triton.testing.do_bench", "warmup_ms": 100, "rep_ms": 300},
        },
        "commands": sorted({str(row.get("command")) for row in all_rows if row.get("command")}),
        "limitations": [
            "The development node exposed more than 24 GiB even though it identified as RTX 4090.",
            "Every formal script enforced a 23552 MiB PyTorch allocator limit before the first CUDA allocation.",
        ],
    }
    (args.public_root / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")


if __name__ == "__main__":
    main()
