"""Finalize speedups, run metadata, and 24 GiB memory evidence."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from .common import upsert_csv_rows, write_json
from .flash_benchmark import FIELDS as FLASH_FIELDS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--run-records", type=Path, required=True)
    return parser.parse_args()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def numeric(row: dict[str, str], field: str) -> float | None:
    value = row.get(field, "")
    if value in {"", None}:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def finalize_speedups(path: Path) -> list[dict[str, str]]:
    rows = read_csv(path)
    eager: dict[tuple[str, str, str], float] = {}
    for row in rows:
        key = (row["seq_len"], row["head_dim"], row["phase"])
        p50 = numeric(row, "p50_ms")
        if (
            row["implementation"] == "eager"
            and row["status"] == "ok"
            and p50 is not None
        ):
            eager[key] = p50
    for row in rows:
        key = (row["seq_len"], row["head_dim"], row["phase"])
        p50 = numeric(row, "p50_ms")
        if row["status"] != "ok" or p50 is None or key not in eager:
            row["speedup_vs_eager"] = ""
        elif row["implementation"] == "eager":
            row["speedup_vs_eager"] = "1.0"
        else:
            row["speedup_vs_eager"] = f"{eager[key] / p50:.9f}"
    upsert_csv_rows(
        path,
        rows,
        key_fields=("implementation", "seq_len", "head_dim", "phase"),
        fieldnames=FLASH_FIELDS,
    )
    return read_csv(path)


def table_memory_summary(
    experiment: str,
    rows: list[dict[str, str]],
) -> dict[str, Any]:
    successful = [row for row in rows if row.get("status") == "ok"]
    allocated = [
        numeric(row, "peak_allocated_mib")
        for row in successful
    ]
    reserved = [numeric(row, "peak_reserved_mib") for row in successful]
    allocated_values = [value for value in allocated if value is not None]
    reserved_values = [value for value in reserved if value is not None]
    return {
        "experiment": experiment,
        "row_count": len(rows),
        "successful_rows": len(successful),
        "oom_rows": sum(row.get("status") == "oom" for row in rows),
        "error_rows": sum(
            row.get("status") not in {"ok", "oom"} for row in rows
        ),
        "max_peak_allocated_mib": (
            max(allocated_values) if allocated_values else None
        ),
        "max_peak_reserved_mib": (
            max(reserved_values) if reserved_values else None
        ),
    }


def build_memory_evidence(
    tables: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    summaries = [
        table_memory_summary(experiment, rows)
        for experiment, rows in tables.items()
    ]
    allocated = [
        summary["max_peak_allocated_mib"]
        for summary in summaries
        if summary["max_peak_allocated_mib"] is not None
    ]
    reserved = [
        summary["max_peak_reserved_mib"]
        for summary in summaries
        if summary["max_peak_reserved_mib"] is not None
    ]
    fractions = {
        float(row["allocator_fraction"])
        for rows in tables.values()
        for row in rows
        if row.get("allocator_fraction")
    }
    limits = {
        int(float(row["allocator_limit_mib"]))
        for rows in tables.values()
        for row in rows
        if row.get("allocator_limit_mib")
    }
    max_allocated = max(allocated) if allocated else None
    max_reserved = max(reserved) if reserved else None
    within_allocator = (
        max_reserved is not None
        and max_reserved <= 23 * 1024
        and limits == {23 * 1024}
    )
    return {
        "schema_version": 1,
        "allocator": {
            "allocator_fraction": (
                next(iter(fractions)) if len(fractions) == 1 else None
            ),
            "allocator_limit_mib": 23 * 1024,
            "observed_allocator_limits_mib": sorted(limits),
        },
        "hard_limit_mib": 24 * 1024,
        "pytorch_peak_allocated_mib": max_allocated,
        "pytorch_peak_reserved_mib": max_reserved,
        "within_24gib": bool(
            within_allocator
            and max_reserved is not None
            and max_reserved <= 24 * 1024
        ),
        "formal_process_summaries": summaries,
        "note": (
            "Aggregate maxima across all successful checkpoint, attention, "
            "compile, and Flash benchmark rows. OOM rows remain in the CSVs."
        ),
    }


def build_run_metadata(
    records: list[dict[str, Any]],
    tables: dict[str, list[dict[str, str]]],
) -> dict[str, Any]:
    experiments: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        experiments[record["experiment"]].append(
            {
                "config_id": record.get("config_id"),
                "command": record["command"],
                "seed": record["seed"],
                "status": record.get("extra", {}).get("status", "ok"),
                "tf32_enabled": record["tf32_enabled"],
                "allocator": record["allocator"],
                "free_memory_mib_at_start": record["hardware"].get(
                    "free_memory_mib_at_start"
                ),
                "timer": record["timer"],
                "warmup": record["warmup"],
                "measurement": record["measurement"],
            }
        )
    first = records[0] if records else {}
    free_values = [
        record["hardware"].get("free_memory_mib_at_start")
        for record in records
        if record.get("hardware", {}).get("free_memory_mib_at_start")
        is not None
    ]
    return {
        "schema_version": 1,
        "assignment": "A2-K",
        "starter_commit": (
            first.get("starter_commit")
            if first
            else "ca8bc81a59b70516f7ebb2da4808daade877c736"
        ),
        "hardware": first.get("hardware", {}),
        "software": first.get("software", {}),
        "formal_process_count": len(records),
        "minimum_free_memory_mib_at_start": (
            min(free_values) if free_values else None
        ),
        "allocator_limit_mib": 23 * 1024,
        "hard_limit_mib": 24 * 1024,
        "timer": (
            "CUDA events for steady-state latency; wall time only for "
            "compile/Triton cold starts"
        ),
        "performance_dtype": "bfloat16",
        "fp32_correctness_tf32_enabled": False,
        "compile": {
            "mode": "reduce-overhead",
            "fresh_process_and_cache_per_formal_compiled_row": True,
            "cold_start_separated": True,
        },
        "table_rows": {
            name: len(rows) for name, rows in tables.items()
        },
        "experiments": dict(sorted(experiments.items())),
    }


def main() -> int:
    args = parse_args()
    flash_rows = finalize_speedups(
        args.results_dir / "flash_benchmark.csv"
    )
    tables = {
        "checkpointing": read_csv(
            args.results_dir / "checkpointing.csv"
        ),
        "attention_baseline": read_csv(
            args.results_dir / "attention_baseline.csv"
        ),
        "compile_comparison": read_csv(
            args.results_dir / "compile_comparison.csv"
        ),
        "flash_benchmark": flash_rows,
    }
    records = json.loads(args.run_records.read_text(encoding="utf-8"))
    write_json(
        args.results_dir / "memory_evidence.json",
        build_memory_evidence(tables),
    )
    write_json(
        args.results_dir / "run_metadata.json",
        build_run_metadata(records, tables),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
