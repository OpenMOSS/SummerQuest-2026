from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from collections.abc import Collection, Iterable
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from profiling.config import read_json, write_json


PROFILE_FIELDS = (
    "run_id",
    "run_name",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "tool",
    "window",
    "stage",
    "event_type",
    "event_name",
    "calls",
    "cpu_self_us",
    "cpu_total_us",
    "cuda_self_us",
    "cuda_total_us",
    "cuda_event_elapsed_us",
    "rank_in_stage",
    "source_run",
)

BENCHMARK_FIELDS = (
    "run_id",
    "run_name",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "warmup_steps",
    "measurement_steps",
    "step_index",
    "time_ms",
    "mean_ms",
    "sample_std_ms",
    "cv",
    "min_ms",
    "median_ms",
    "max_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "last_loss",
    "status",
    "source_run",
)

MEMORY_FIELDS = (
    "run_id",
    "requested_model",
    "requested_context",
    "requested_batch",
    "actual_model",
    "actual_context",
    "actual_batch",
    "mode",
    "dtype",
    "status",
    "failure_stage",
    "peak_active_mib",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "largest_allocation_mib",
    "fallback_reason",
    "source_run",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create lightweight public A2-P summaries")
    parser.add_argument(
        "--section",
        choices=("benchmark", "profile", "mixed", "memory", "all"),
        default="all",
    )
    parser.add_argument("--results-root", type=Path, default=Path("results"))
    parser.add_argument("--public-root", type=Path, default=Path("results/public"))
    parser.add_argument("--top-k", type=int, default=10)
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args()


def _write_csv(path: Path, fieldnames: Collection[str], rows: Iterable[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def _json_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*.json") if path.is_file())


def _public_benchmark_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "run_id",
        "run_name",
        "status",
        "command",
        "started_at",
        "finished_at",
        "model_size",
        "model_config",
        "parameter_count",
        "model_fingerprint",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "autocast_dtype",
        "warmup_steps",
        "measurement_steps",
        "seed",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "timer",
        "synchronize_before_and_after_step",
        "attention_instrumentation",
        "output_file",
        "environment",
    )
    return {key: payload.get(key) for key in keep}


def summarize_benchmark(input_root: Path, output: Path, metadata_output: Path, strict: bool) -> int:
    results = []
    for path in _json_files(input_root):
        payload = read_json(path)
        if payload.get("kind") != "benchmark_result":
            continue
        results.append((path, payload))

    if strict and len(results) < 4:
        raise ValueError(f"expected at least 4 benchmark runs, found {len(results)}")

    rows = []
    for path, payload in results:
        timings = payload.get("timings_ms", [])
        summary = payload.get("summary", {})
        if not timings:
            timings = [None]
        for index, timing in enumerate(timings):
            rows.append(
                {
                    "run_id": payload.get("run_id"),
                    "run_name": payload.get("run_name"),
                    "model_size": payload.get("model_size"),
                    "batch_size": payload.get("batch_size"),
                    "context_length": payload.get("context_length"),
                    "mode": payload.get("mode"),
                    "dtype": payload.get("dtype"),
                    "warmup_steps": payload.get("warmup_steps"),
                    "measurement_steps": payload.get("measurement_steps"),
                    "step_index": index if timing is not None else None,
                    "time_ms": timing,
                    "mean_ms": summary.get("mean_ms"),
                    "sample_std_ms": summary.get("sample_std_ms"),
                    "cv": summary.get("cv"),
                    "min_ms": summary.get("min_ms"),
                    "median_ms": summary.get("median_ms"),
                    "max_ms": summary.get("max_ms"),
                    "peak_allocated_mib": payload.get("peak_allocated_mib"),
                    "peak_reserved_mib": payload.get("peak_reserved_mib"),
                    "last_loss": payload.get("last_loss"),
                    "status": payload.get("status"),
                    "source_run": path.name,
                }
            )
    _write_csv(output, BENCHMARK_FIELDS, rows)
    write_json(metadata_output, [_public_benchmark_metadata(payload) for _, payload in results])
    return len(results)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _public_profile_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "run_id",
        "run_name",
        "status",
        "failure_stage",
        "error_type",
        "error_summary",
        "command",
        "started_at",
        "finished_at",
        "model_size",
        "model_config",
        "parameter_count",
        "model_fingerprint",
        "batch_size",
        "context_length",
        "mode",
        "dtype",
        "autocast_dtype",
        "warmup_steps",
        "measurement_steps",
        "seed",
        "optimizer",
        "learning_rate",
        "weight_decay",
        "tool",
        "tool_config",
        "trace_file",
        "stage_markers",
        "trace_validation",
        "cuda_event_stage_elapsed_us",
        "environment",
    )
    return {key: payload.get(key) for key in keep}


def summarize_profile(
    input_root: Path,
    output_csv: Path,
    output_metadata: Path,
    top_k: int,
    strict: bool,
) -> int:
    metadata_files = sorted(input_root.glob("*.metadata.json"))
    payloads = [read_json(path) for path in metadata_files]
    payloads = [payload for payload in payloads if payload.get("kind") == "compute_profile_result"]
    if strict and len(payloads) != 6:
        raise ValueError(f"expected exactly 6 compute profile runs, found {len(payloads)}")

    rows: list[dict[str, Any]] = []
    for payload in payloads:
        run_name = payload["run_name"]
        common = {
            "run_id": payload.get("run_id"),
            "run_name": run_name,
            "model_size": payload.get("model_size"),
            "batch_size": payload.get("batch_size"),
            "context_length": payload.get("context_length"),
            "mode": payload.get("mode"),
            "dtype": payload.get("dtype"),
            "tool": payload.get("tool"),
            "window": "profile/measure",
            "source_run": f"{run_name}.metadata.json",
        }
        sources = {
            "stage": input_root / f"{run_name}.stages.csv",
            "op": input_root / f"{run_name}.operators.csv",
            "kernel": input_root / f"{run_name}.kernels.csv",
        }
        for event_type, source in sources.items():
            if not source.is_file():
                continue
            source_rows = _read_csv(source)
            if event_type == "stage":
                source_rows = [row for row in source_rows if row.get("window") == "profile/measure" and row.get("device_type") == "cpu"]
                aggregated_stages: dict[tuple[str, str], dict[str, Any]] = {}
                for row in source_rows:
                    key = (row.get("stage", "profile/measure"), row.get("event_name", ""))
                    target = aggregated_stages.setdefault(
                        key,
                        {
                            "window": "profile/measure",
                            "stage": key[0],
                            "event_type": "stage",
                            "event_name": key[1],
                            "calls": 0,
                            "cpu_self_us": 0.0,
                            "cpu_total_us": 0.0,
                            "cuda_self_us": 0.0,
                            "cuda_total_us": 0.0,
                            "cuda_event_elapsed_us": 0.0,
                        },
                    )
                    for field in (
                        "calls",
                        "cpu_self_us",
                        "cpu_total_us",
                        "cuda_self_us",
                        "cuda_total_us",
                        "cuda_event_elapsed_us",
                    ):
                        target[field] += float(row.get(field) or 0.0)
                source_rows = list(aggregated_stages.values())
            grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
            for row in source_rows:
                grouped[row.get("stage", "profile/measure")].append(row)
            for stage, stage_items in grouped.items():
                stage_items.sort(
                    key=lambda row: (
                        -max(
                            float(row.get("cuda_total_us") or 0.0),
                            float(row.get("cpu_total_us") or 0.0),
                        )
                    )
                )
                selected = stage_items if event_type == "stage" else stage_items[:top_k]
                for rank, item in enumerate(selected, start=1):
                    rows.append(
                        {
                            **common,
                            "stage": stage,
                            "event_type": event_type,
                            "event_name": item.get("event_name"),
                            "calls": item.get("calls"),
                            "cpu_self_us": item.get("cpu_self_us"),
                            "cpu_total_us": item.get("cpu_total_us"),
                            "cuda_self_us": item.get("cuda_self_us"),
                            "cuda_total_us": item.get("cuda_total_us"),
                            "cuda_event_elapsed_us": item.get("cuda_event_elapsed_us"),
                            "rank_in_stage": rank,
                        }
                    )
    _write_csv(output_csv, PROFILE_FIELDS, rows)
    write_json(output_metadata, [_public_profile_metadata(payload) for payload in payloads])
    return len(payloads)


def summarize_mixed(input_root: Path, output: Path, strict: bool) -> int:
    sections: dict[str, Any] = {
        "schema_version": "a2p-mixed-precision-v1",
        "accumulation": [],
        "toy_model": [],
        "benchmarks": [],
    }
    count = 0
    for path in _json_files(input_root):
        payload = read_json(path)
        kind = payload.get("kind")
        if kind == "mixed_precision_accumulation":
            sections["accumulation"].append(payload)
        elif kind == "mixed_precision_toy_model":
            sections["toy_model"].append(payload)
        elif kind == "benchmark_result" and payload.get("dtype") in {"fp32", "bf16"}:
            sections["benchmarks"].append(payload)
        else:
            continue
        count += 1
    if strict:
        if not sections["accumulation"] or not sections["toy_model"] or len(sections["benchmarks"]) < 6:
            raise ValueError("mixed precision summary is missing accumulation, ToyModel, or six benchmark runs")
    write_json(output, sections)
    return count


def _public_memory_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "run_id",
        "run_name",
        "status",
        "failure_stage",
        "error_type",
        "error_summary",
        "command",
        "started_at",
        "finished_at",
        "requested_model",
        "requested_context",
        "requested_batch",
        "actual_model",
        "actual_context",
        "actual_batch",
        "model_config",
        "parameter_count",
        "model_fingerprint",
        "mode",
        "dtype",
        "warmup_steps",
        "measurement_steps",
        "seed",
        "max_entries",
        "fallback_reason",
        "fallback_parent_run_id",
        "fallback_attempt",
        "peak_active_mib",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "ending_allocated_mib",
        "ending_reserved_mib",
        "largest_allocation_mib",
        "largest_segment_allocation_mib",
        "largest_tensor_allocation",
        "environment",
    )
    public = {key: payload.get(key) for key in keep}
    allocation = public.get("largest_tensor_allocation")
    if isinstance(allocation, dict):
        sanitized = dict(allocation)
        frames = []
        for frame in allocation.get("frames", []):
            if not isinstance(frame, dict):
                continue
            filename = str(frame.get("file") or "")
            if "site-packages/" in filename:
                filename = filename.split("site-packages/", 1)[1]
            if filename.startswith(".venv/"):
                filename = Path(filename).name
            frames.append({"file": filename or None, "line": frame.get("line"), "name": frame.get("name")})
        sanitized["frames"] = frames
        public["largest_tensor_allocation"] = sanitized
    return public


def _public_saved_tensor_metadata(payload: dict[str, Any]) -> dict[str, Any]:
    keep = (
        "schema_version",
        "run_id",
        "run_name",
        "kind",
        "status",
        "failure_stage",
        "error_type",
        "error_summary",
        "started_at",
        "finished_at",
        "model_size",
        "model_config",
        "batch_size",
        "context_length",
        "dtype",
        "seed",
        "saved_tensor_count",
        "parameter_storage_record_count",
        "activation_saved_tensor_count",
        "unique_activation_storage_count",
        "logical_activation_saved_bytes",
        "logical_activation_saved_mib",
        "unique_activation_saved_bytes",
        "unique_activation_saved_mib",
        "top_activation_logical_groups",
        "top_unique_activation_storages",
        "parameter_bytes",
        "parameter_mib",
        "parameter_gradient_bytes",
        "parameter_gradient_mib",
        "missing_parameter_gradients",
        "input_gradient_bytes",
        "input_gradient_mib",
        "total_produced_gradient_bytes",
        "total_produced_gradient_mib",
        "environment",
    )
    return {key: payload.get(key) for key in keep}


def summarize_memory(
    input_root: Path,
    output_csv: Path,
    output_metadata: Path,
    strict: bool,
) -> int:
    payloads = []
    saved_tensor_payloads = []
    for path in _json_files(input_root):
        payload = read_json(path)
        if payload.get("kind") == "memory_profile_result":
            payloads.append((path, payload))
        elif payload.get("kind") == "saved_tensor_diagnostic":
            saved_tensor_payloads.append(payload)
    if strict and len(payloads) < 4:
        raise ValueError(f"expected at least 4 memory runs, found {len(payloads)}")

    rows = []
    for path, payload in payloads:
        rows.append(
            {
                "run_id": payload.get("run_id"),
                "requested_model": payload.get("requested_model"),
                "requested_context": payload.get("requested_context"),
                "requested_batch": payload.get("requested_batch"),
                "actual_model": payload.get("actual_model"),
                "actual_context": payload.get("actual_context"),
                "actual_batch": payload.get("actual_batch"),
                "mode": payload.get("mode"),
                "dtype": payload.get("dtype"),
                "status": payload.get("status"),
                "failure_stage": payload.get("failure_stage"),
                "peak_active_mib": payload.get("peak_active_mib"),
                "peak_allocated_mib": payload.get("peak_allocated_mib"),
                "peak_reserved_mib": payload.get("peak_reserved_mib"),
                "largest_allocation_mib": payload.get("largest_allocation_mib"),
                "fallback_reason": payload.get("fallback_reason"),
                "source_run": path.name,
            }
        )
    _write_csv(output_csv, MEMORY_FIELDS, rows)
    write_json(
        output_metadata,
        [_public_memory_metadata(payload) for _, payload in payloads] + [_public_saved_tensor_metadata(payload) for payload in saved_tensor_payloads],
    )
    return len(payloads)


def main() -> int:
    args = parse_args()
    results_root = args.results_root
    public_root = args.public_root
    counts: dict[str, int] = {}

    if args.section in {"benchmark", "all"}:
        counts["benchmark"] = summarize_benchmark(
            results_root / "benchmark" / "raw",
            public_root / "benchmark.csv",
            public_root / "benchmark" / "run_metadata.json",
            args.strict,
        )
    if args.section in {"profile", "all"}:
        counts["profile"] = summarize_profile(
            results_root / "profile" / "raw",
            public_root / "profile" / "trace_summary.csv",
            public_root / "profile" / "run_metadata.json",
            args.top_k,
            args.strict,
        )
    if args.section in {"mixed", "all"}:
        counts["mixed"] = summarize_mixed(
            results_root / "mixed_precision" / "raw",
            public_root / "mixed_precision.json",
            args.strict,
        )
    if args.section in {"memory", "all"}:
        counts["memory"] = summarize_memory(
            results_root / "memory" / "raw",
            public_root / "memory" / "peaks.csv",
            public_root / "memory" / "run_metadata.json",
            args.strict,
        )

    print(json.dumps(counts, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
