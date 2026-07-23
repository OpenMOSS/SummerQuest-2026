#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.io_utils import artifact_name, upsert_csv_rows


PHASE_NAMES = {
    "profile/warmup",
    "profile/measure",
    "forward",
    "loss",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize A2-P profiler artifacts")
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="summarize one Chrome trace")
    trace.add_argument("--trace", type=Path, required=True)
    trace.add_argument("--metadata", type=Path, required=True)
    trace.add_argument("--run-id", required=True)
    trace.add_argument("--output", type=Path, required=True)
    trace.add_argument("--operator-summary", type=Path)
    trace.add_argument("--top-ops", type=int, default=25)
    trace.add_argument("--top-kernels", type=int, default=25)

    validate = subparsers.add_parser("validate-profile-matrix", help="validate the required 2 x 3 trace matrix")
    validate.add_argument("--metadata", type=Path, required=True)

    benchmark = subparsers.add_parser("validate-benchmark", help="validate the required benchmark runs")
    benchmark.add_argument("--csv", type=Path, required=True)

    mixed = subparsers.add_parser("validate-mixed-precision", help="validate mixed_precision.json")
    mixed.add_argument("--json", type=Path, required=True)

    memory = subparsers.add_parser("validate-memory", help="validate the required memory matrix and fallbacks")
    memory.add_argument("--peaks", type=Path, required=True)
    return parser.parse_args()


def load_metadata_record(path: Path, run_id: str) -> dict[str, Any]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON list in {path}")
    matches = [record for record in records if record.get("run_id") == run_id]
    if len(matches) != 1:
        raise ValueError(f"expected exactly one metadata record for {run_id}, found {len(matches)}")
    return matches[0]


def trace_events(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        events = payload.get("traceEvents", [])
    elif isinstance(payload, list):
        events = payload
    else:
        raise ValueError(f"unsupported trace format in {path}")
    return [event for event in events if isinstance(event, dict)]


def event_kind(event: dict[str, Any]) -> str | None:
    category = str(event.get("cat", "")).lower()
    name = str(event.get("name", ""))
    if name in PHASE_NAMES or "user_annotation" in category:
        return "phase_range"
    if "kernel" in category:
        return "cuda_kernel"
    if "cpu_op" in category:
        return "cpu_op"
    return None


def aggregate_events(events: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, float]]]:
    aggregated: dict[str, dict[str, dict[str, float]]] = defaultdict(lambda: defaultdict(lambda: {"calls": 0.0, "duration_us": 0.0}))
    for event in events:
        if event.get("ph") != "X":
            continue
        kind = event_kind(event)
        if kind is None:
            continue
        name = str(event.get("name", "<unnamed>"))
        try:
            duration = float(event.get("dur", 0.0))
        except (TypeError, ValueError):
            duration = 0.0
        aggregated[kind][name]["calls"] += 1
        aggregated[kind][name]["duration_us"] += duration
    return aggregated


def select_rows(
    aggregated: dict[str, dict[str, dict[str, float]]],
    *,
    top_ops: int,
    top_kernels: int,
) -> list[tuple[str, str, dict[str, float]]]:
    selected: list[tuple[str, str, dict[str, float]]] = []
    for name, values in sorted(aggregated.get("phase_range", {}).items()):
        if name in PHASE_NAMES:
            selected.append(("phase_range", name, values))
    for kind, limit in (("cpu_op", top_ops), ("cuda_kernel", top_kernels)):
        ranked = sorted(
            aggregated.get(kind, {}).items(),
            key=lambda item: item[1]["duration_us"],
            reverse=True,
        )[:limit]
        selected.extend((kind, name, values) for name, values in ranked)
    return selected


def select_operator_rows(path: Path, *, top_ops: int) -> list[dict[str, Any]]:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON list in {path}")
    coalesced: dict[str, dict[str, Any]] = {}
    for record in records:
        name = str(record.get("name", "<unnamed>"))
        current = coalesced.setdefault(
            name,
            {
                "name": name,
                "calls": 0,
                "cpu_total_us": 0.0,
                "cpu_self_us": 0.0,
                "device_total_us": 0.0,
                "device_self_us": 0.0,
            },
        )
        for field in (
            "calls",
            "cpu_total_us",
            "cpu_self_us",
            "device_total_us",
            "device_self_us",
        ):
            current[field] = max(float(current[field]), float(record.get(field, 0.0)))
    normalized = list(coalesced.values())
    phases = [record for record in normalized if record["name"] in PHASE_NAMES]
    operations = [record for record in normalized if record["name"] not in PHASE_NAMES]
    operations.sort(key=lambda record: float(record.get("device_total_us", 0.0)), reverse=True)
    if not any(float(record.get("device_total_us", 0.0)) > 0 for record in operations):
        operations.sort(key=lambda record: float(record.get("cpu_total_us", 0.0)), reverse=True)
    return [*sorted(phases, key=lambda record: record["name"]), *operations[:top_ops]]


def summarize_trace(args: argparse.Namespace) -> int:
    metadata = load_metadata_record(args.metadata, args.run_id)
    config = metadata["configuration"]
    aggregated = aggregate_events(trace_events(args.trace))
    selected_trace = select_rows(
        aggregated,
        top_ops=0 if args.operator_summary is not None else args.top_ops,
        top_kernels=args.top_kernels,
    )
    selected_operators = select_operator_rows(args.operator_summary, top_ops=args.top_ops) if args.operator_summary is not None else []
    if not selected_trace and not selected_operators:
        raise ValueError(f"no CPU ops, CUDA kernels, or phase ranges found in {args.trace}")

    rows: list[dict[str, Any]] = []
    for record_type, name, values in selected_trace:
        if args.operator_summary is not None and record_type == "phase_range":
            continue
        duration = values["duration_us"]
        rows.append(
            {
                "run_id": args.run_id,
                "model_size": config["model_size"],
                "context_length": config["context_length"],
                "batch_size": config["batch_size"],
                "mode": config["mode"],
                "dtype": config["dtype"],
                "tool": metadata["profiler"],
                "record_type": record_type,
                "name": name,
                "calls": int(values["calls"]),
                "cpu_total_us": duration if record_type != "cuda_kernel" else 0.0,
                "cuda_total_us": duration if record_type == "cuda_kernel" else 0.0,
                "trace_file": artifact_name(args.trace),
                "command": metadata["command"],
            }
        )
    for record in selected_operators:
        name = str(record["name"])
        rows.append(
            {
                "run_id": args.run_id,
                "model_size": config["model_size"],
                "context_length": config["context_length"],
                "batch_size": config["batch_size"],
                "mode": config["mode"],
                "dtype": config["dtype"],
                "tool": metadata["profiler"],
                "record_type": "phase_range" if name in PHASE_NAMES else "cpu_op",
                "name": name,
                "calls": int(record.get("calls", 0)),
                "cpu_total_us": float(record.get("cpu_total_us", 0.0)),
                "cuda_total_us": float(record.get("device_total_us", 0.0)),
                "trace_file": artifact_name(args.trace),
                "command": metadata["command"],
            }
        )
    upsert_csv_rows(args.output, rows)
    print(f"wrote {len(rows)} rows for {args.run_id} to {args.output}")
    return 0


def is_power_of_two(value: int) -> bool:
    return value > 0 and value & (value - 1) == 0


def validate_profile_matrix(path: Path) -> int:
    records = json.loads(path.read_text())
    if not isinstance(records, list):
        raise ValueError(f"expected a JSON list in {path}")
    successful = [record for record in records if record.get("status") == "success"]
    if len(successful) != 6:
        raise ValueError(f"expected 6 successful profile runs, found {len(successful)}")

    configurations = [record["configuration"] for record in successful]
    models = {config["model_size"] for config in configurations}
    contexts = {int(config["context_length"]) for config in configurations}
    if len(models) != 2:
        raise ValueError(f"expected 2 model sizes, found {sorted(models)}")
    if len(contexts) != 3:
        raise ValueError(f"expected 3 context lengths, found {sorted(contexts)}")
    if any(context <= 128 or not is_power_of_two(context) for context in contexts):
        raise ValueError("all context lengths must be powers of two greater than 128")
    expected = {(model, context) for model in models for context in contexts}
    actual = {(config["model_size"], int(config["context_length"])) for config in configurations}
    if actual != expected:
        raise ValueError(f"profile matrix is incomplete: expected {expected}, found {actual}")
    for record, config in zip(successful, configurations, strict=True):
        if record.get("profiler") != "torch":
            raise ValueError("all six runs must use torch.profiler")
        if config["mode"] != "train_step" or int(config["steps"]) != 1:
            raise ValueError("all six runs must capture exactly one train_step")
        if int(config["warmup"]) < 1:
            raise ValueError("each profile run must include at least one warm-up step")
    print(f"valid profile matrix: models={sorted(models)} contexts={sorted(contexts)} runs={len(successful)}")
    return 0


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if not rows:
        raise ValueError(f"no data rows found in {path}")
    return rows


def validate_benchmark(path: Path) -> int:
    rows = read_csv_rows(path)
    groups: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        groups[row["run_id"]].append(row)

    valid_runs: list[dict[str, str]] = []
    for run_id, run_rows in groups.items():
        first = run_rows[0]
        expected_steps = int(first["steps"])
        if len(run_rows) != expected_steps:
            raise ValueError(f"{run_id} has {len(run_rows)} timings, expected {expected_steps}")
        measurement_steps = {int(row["measurement_step"]) for row in run_rows}
        if measurement_steps != set(range(expected_steps)):
            raise ValueError(f"{run_id} has incomplete measurement_step values")
        timings = [float(row["time_ms"]) for row in run_rows]
        mean = statistics.mean(timings)
        sample_std = statistics.stdev(timings) if len(timings) > 1 else 0.0
        cv = sample_std / mean if not math.isclose(mean, 0.0) else 0.0
        for field, expected in (("mean_ms", mean), ("sample_std_ms", sample_std), ("cv", cv)):
            if not math.isclose(float(first[field]), expected, rel_tol=1e-9, abs_tol=1e-9):
                raise ValueError(f"{run_id} has an inconsistent {field}")
        valid_runs.append(first)

    formal = [
        row
        for row in valid_runs
        if row["model_size"] == "small" and int(row["batch_size"]) == 4 and int(row["context_length"]) == 512 and row["dtype"] == "fp32" and int(row["steps"]) >= 10
    ]
    for mode in ("forward", "forward_backward", "train_step"):
        if not any(row["mode"] == mode and int(row["warmup"]) >= 5 for row in formal):
            raise ValueError(f"missing small/bs4/ctx512/fp32 {mode} run with warmup >= 5")
    train_warmups = {int(row["warmup"]) for row in formal if row["mode"] == "train_step"}
    if not {0, 5}.issubset(train_warmups):
        raise ValueError("train_step warm-up comparison must include warmup 0 and 5")
    print(f"valid benchmark results: runs={len(valid_runs)} formal_runs={len(formal)}")
    return 0


def validate_mixed_precision(path: Path) -> int:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict):
        raise ValueError(f"expected a JSON object in {path}")
    accumulation = payload.get("accumulation", {}).get("results", {})
    expected_accumulations = {
        "fp32_input_fp32_accumulator",
        "fp16_input_fp16_accumulator",
        "fp16_input_fp32_accumulator_implicit",
        "fp16_input_fp32_accumulator_explicit",
    }
    if set(accumulation) != expected_accumulations:
        raise ValueError("mixed-precision accumulation must contain all four fixed cases")
    toy = payload.get("toy_model", {})
    if toy.get("configuration", {}).get("autocast_dtype") != "bfloat16":
        raise ValueError("ToyModel must use BF16 autocast")
    required_dtypes = {
        "parameters_inside_autocast",
        "fc1_output",
        "layer_norm_output",
        "logits",
        "loss",
        "gradients",
    }
    if not required_dtypes.issubset(toy.get("dtypes", {})):
        raise ValueError("ToyModel dtype evidence is incomplete")
    benchmarks = payload.get("benchmarks", [])
    if not benchmarks:
        raise ValueError("at least one FP32/BF16 benchmark comparison is required")
    for record in benchmarks:
        expected_steps = int(record["configuration"]["steps"])
        for dtype in ("fp32", "bf16"):
            result = record[dtype]
            if len(result["timings_ms"]) != expected_steps:
                raise ValueError(f"{record['run_id']} has incomplete {dtype} timings")
            if not result["numerics"]["logits_finite"]:
                raise ValueError(f"{record['run_id']} produced non-finite {dtype} logits")
        if record.get("comparison", {}).get("sampled_logits_count", 0) < 1:
            raise ValueError(f"{record['run_id']} has no numerical comparison sample")
    print(f"valid mixed-precision results: benchmarks={len(benchmarks)}")
    return 0


def validate_memory(path: Path) -> int:
    rows = read_csv_rows(path)
    indexed = {(row["model_size"], int(row["context_length"]), row["mode"]): row for row in rows}
    for row in rows:
        for field in (
            "active_peak_bytes",
            "allocated_peak_bytes",
            "reserved_peak_bytes",
            "residual_stream_theoretical_bytes",
            "largest_allocation_bytes",
        ):
            if int(row[field]) < 0:
                raise ValueError(f"{row['run_id']} has a negative {field}")

    for mode in ("forward", "train_step"):
        context_128 = indexed.get(("xl", 128, mode))
        context_2048 = indexed.get(("xl", 2048, mode))
        if context_128 is None:
            raise ValueError(f"missing XL/context 128/{mode} memory attempt")
        if context_128["status"] not in {"success", "oom"}:
            raise ValueError(f"XL/context 128/{mode} did not record success or OOM")
        if context_2048 is None:
            raise ValueError(f"missing XL/context 2048/{mode} memory attempt")
        if context_2048["status"] == "success":
            continue
        if context_2048["status"] != "oom":
            raise ValueError(f"XL/context 2048/{mode} failed without an OOM record")
        xl_1024 = indexed.get(("xl", 1024, mode))
        if xl_1024 is None:
            raise ValueError(f"OOM fallback must try XL/context 1024/{mode} first")
        if xl_1024["status"] == "success":
            continue
        if xl_1024["status"] != "oom":
            raise ValueError(f"XL/context 1024/{mode} fallback did not record success or OOM")
        large_2048 = indexed.get(("large", 2048, mode))
        if large_2048 is None or large_2048["status"] != "success":
            raise ValueError(f"OOM fallback must finish with successful Large/context 2048/{mode}")
    print(f"valid memory results: runs={len(rows)}")
    return 0


def main() -> int:
    args = parse_args()
    if args.command == "trace":
        return summarize_trace(args)
    if args.command == "validate-profile-matrix":
        return validate_profile_matrix(args.metadata)
    if args.command == "validate-benchmark":
        return validate_benchmark(args.csv)
    if args.command == "validate-mixed-precision":
        return validate_mixed_precision(args.json)
    if args.command == "validate-memory":
        return validate_memory(args.peaks)
    raise AssertionError(f"unexpected command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
