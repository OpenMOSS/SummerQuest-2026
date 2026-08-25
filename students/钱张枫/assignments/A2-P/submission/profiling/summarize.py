#!/usr/bin/env python3
"""Create compact, public-safe summaries from local profiling artifacts.

The profiler's Chrome traces, Nsight reports, and CUDA memory snapshots stay
local.  This script consumes their already-exported tabular summaries and writes
only the small CSV/JSON files referenced by the A2-P report.

Examples
--------

Consolidate one or more ``torch.profiler`` / benchmark summaries::

    uv run python profiling/summarize.py trace \
        --input results/profile/raw/*.csv \
        --output results/profile/trace_summary.csv

Normalize an existing memory peak table after reviewing it for publication::

    uv run python profiling/summarize.py memory \
        --input results/memory/peaks.csv \
        --output results/memory/peaks_public.csv

The commands deliberately store only input basenames in metadata.  Do not pass a
raw Chrome trace, ``.nsys-rep``, or memory ``.pickle`` file to this program.
"""

from __future__ import annotations

import argparse
import csv
import datetime as datetime_module
import json
import math
import os
import re
import shlex
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any
from collections.abc import Iterable, Mapping, Sequence


TRACE_COLUMNS = (
    "run_name",
    "stage",
    "op_name",
    "calls",
    "cpu_self_time_us",
    "cpu_total_time_us",
    "cuda_self_time_us",
    "cuda_total_time_us",
    "source_file",
)

MEMORY_PREFERRED_COLUMNS = (
    "run_id",
    "timestamp_utc",
    "status",
    "requested_model_size",
    "requested_context_length",
    "requested_batch_size",
    "requested_mode",
    "model_size",
    "context_length",
    "batch_size",
    "mode",
    "dtype",
    "seed",
    "warmup_steps",
    "fallback_level",
    "fallback_reason",
    "measurement_elapsed_ms",
    "memory_history_status",
    "torch_profiler_memory_enabled",
    "torch_profiler_memory_status",
    "snapshot_saved",
    "snapshot_file",
    "profiler_trace_file",
    "active_bytes_after_step",
    "active_mib_after_step",
    "allocated_bytes_after_step",
    "allocated_mib_after_step",
    "reserved_bytes_after_step",
    "reserved_mib_after_step",
    "active_bytes_peak",
    "active_mib_peak",
    "allocated_bytes_peak",
    "allocated_mib_peak",
    "reserved_bytes_peak",
    "reserved_mib_peak",
    "peak_allocated_bytes",
    "peak_allocated_mib",
    "peak_reserved_bytes",
    "peak_reserved_mib",
    "failure_stage",
    "exception_type",
    "error_message",
    "warning_stages",
    "warning_message",
)

FORBIDDEN_SUFFIXES = {".pickle", ".pkl", ".nsys-rep", ".sqlite", ".zip", ".pt", ".pth"}
SENSITIVE_COLUMN_TOKENS = {
    "hostname",
    "host",
    "username",
    "user",
    "home",
    "directory",
    "path",
    "ip",
    "uuid",
    "pid",
    "process",
    "token",
    "secret",
    "password",
    "cookie",
}


def utc_timestamp() -> str:
    return datetime_module.datetime.now(datetime_module.timezone.utc).isoformat(timespec="milliseconds").replace("+00:00", "Z")


def normalize_name(value: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def redact_text(value: object, limit: int = 500) -> str:
    """Remove accidental public identifiers from errors and free-form CSV cells."""

    text = " ".join(str(value).split())
    text = re.sub(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", "<ip>", text)
    text = re.sub(
        r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b",
        "<id>",
        text,
    )
    text = re.sub(r"(?<![\w<])/(?:[^\s:'\"()\[\],]+/)*[^\s:'\"()\[\],]+", "<path>", text)
    return text[:limit]


def positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="汇总本地 profiler 的轻量 CSV；不复制 raw trace/snapshot。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    trace = subparsers.add_parser("trace", help="合并/规范化 torch.profiler 或 nsys 导出的操作时间汇总")
    trace.add_argument("--input", type=Path, nargs="+", required=True, help="CSV 文件或仅含 CSV 的目录")
    trace.add_argument("--output", type=Path, default=Path("results") / "profile" / "trace_summary.csv")
    trace.add_argument("--metadata", type=Path, default=Path("results") / "profile" / "run_metadata.json")
    trace.add_argument("--top", type=positive_int, default=200, help="每个 run/stage 保留的最大累计 CUDA 时间操作数")
    trace.add_argument("--run-name", default=None, help="仅当输入 CSV 缺少 run_name 时的安全标签")

    memory = subparsers.add_parser("memory", help="去除敏感列并规范化轻量 peaks CSV")
    memory.add_argument("--input", type=Path, required=True, help="memory_snapshot.py 写出的 peaks.csv")
    memory.add_argument("--output", type=Path, default=Path("results") / "memory" / "peaks_public.csv")
    memory.add_argument("--metadata", type=Path, default=Path("results") / "memory" / "summary_metadata.json")
    memory.add_argument(
        "--include-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="保留 OOM/error 行，确保 fallback 与失败配置可追溯",
    )
    return parser.parse_args(argv)


def atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def atomic_write_csv(path: Path, fieldnames: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f"{path.suffix}.tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def validate_input_path(path: Path) -> None:
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise ValueError(f"拒绝读取大型或禁止提交的原始 artifact：{path.name}")
    if path.suffix.lower() == ".json":
        raise ValueError(f"拒绝读取 Chrome trace/JSON；请先导出轻量 CSV：{path.name}")


def expand_csv_inputs(inputs: Sequence[Path], output: Path) -> list[Path]:
    files: list[Path] = []
    output_resolved = output.resolve()
    for input_path in inputs:
        validate_input_path(input_path)
        if input_path.is_file():
            candidates = [input_path]
        elif input_path.is_dir():
            candidates = sorted(candidate for candidate in input_path.rglob("*.csv") if candidate.is_file())
        else:
            raise FileNotFoundError(f"找不到输入：{input_path}")
        for candidate in candidates:
            validate_input_path(candidate)
            if candidate.resolve() == output_resolved:
                continue
            if candidate not in files:
                files.append(candidate)
    if not files:
        raise ValueError("没有可汇总的 CSV 输入（输出文件会被自动排除）")
    return files


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise ValueError(f"CSV 没有 header：{path.name}")
        fieldnames = [str(name) for name in reader.fieldnames]
        return fieldnames, [dict(row) for row in reader]


def header_lookup(fieldnames: Sequence[str]) -> dict[str, str]:
    return {normalize_name(field): field for field in fieldnames}


def first_value(row: Mapping[str, str], lookup: Mapping[str, str], aliases: Sequence[str], default: str = "") -> str:
    for alias in aliases:
        actual = lookup.get(normalize_name(alias))
        if actual is not None:
            value = row.get(actual)
            if value is not None and str(value).strip():
                return str(value).strip()
    return default


def parse_count(value: str) -> int:
    compact = value.strip().replace(",", "")
    if not compact or compact.lower() in {"na", "n/a", "none", "-", "--"}:
        return 0
    try:
        return int(float(compact))
    except ValueError:
        return 0


def parse_microseconds(value: str) -> float:
    """Parse raw microseconds or common profiler strings such as ``1.24ms``."""

    compact = value.strip().replace(",", "")
    if not compact or compact.lower() in {"na", "n/a", "none", "-", "--"}:
        return 0.0
    match = re.fullmatch(r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(ns|us|µs|ms|s)?", compact, flags=re.IGNORECASE)
    if match is None:
        return 0.0
    number = float(match.group(1))
    if not math.isfinite(number):
        return 0.0
    unit = (match.group(2) or "us").lower()
    multiplier = {"ns": 0.001, "us": 1.0, "µs": 1.0, "ms": 1_000.0, "s": 1_000_000.0}[unit]
    return number * multiplier


def infer_stage(explicit_stage: str, operation: str) -> str:
    if explicit_stage:
        return explicit_stage
    name = operation.lower()
    if "profile/warmup" in name or "warmup" in name:
        return "profile/warmup"
    if "profile/measure" in name or "measure" in name:
        return "profile/measure"
    if "attention/scores" in name or "score" in name and "attention" in name:
        return "attention/scores"
    if "attention/softmax" in name or "softmax" in name:
        return "attention/softmax"
    if "attention/value" in name or "attention" in name and ("value" in name or "matmul" in name):
        return "attention/value"
    if "backward" in name or "autograd" in name:
        return "backward"
    if "optimizer" in name or "adam" in name or "zero_grad" in name:
        return "optimizer"
    if "forward" in name:
        return "forward"
    return "unlabelled"


def trace_row_from_source(
    row: Mapping[str, str],
    lookup: Mapping[str, str],
    source: Path,
    fallback_run_name: str | None,
) -> dict[str, Any] | None:
    operation = first_value(
        row,
        lookup,
        ("op_name", "operator", "op", "kernel_name", "kernel", "name", "key"),
    )
    if not operation:
        return None
    run_name = first_value(row, lookup, ("run_name", "run", "profile_name", "trace_name"), fallback_run_name or source.stem)
    explicit_stage = first_value(row, lookup, ("stage", "stage_range", "range", "nvtx_range"))
    return {
        "run_name": redact_text(run_name, 120),
        "stage": redact_text(infer_stage(explicit_stage, operation), 120),
        "op_name": redact_text(operation, 300),
        "calls": parse_count(first_value(row, lookup, ("calls", "count", "number_of_calls"))),
        "cpu_self_time_us": parse_microseconds(
            first_value(row, lookup, ("cpu_self_time_us", "self_cpu_time_total", "self_cpu_time_us", "cpu_self_time", "self_cpu_time"))
        ),
        "cpu_total_time_us": parse_microseconds(
            first_value(row, lookup, ("cpu_total_time_us", "cpu_time_total", "cpu_total_time", "cpu_time"))
        ),
        "cuda_self_time_us": parse_microseconds(
            first_value(row, lookup, ("cuda_self_time_us", "self_cuda_time_total", "self_cuda_time_us", "cuda_self_time", "self_cuda_time"))
        ),
        "cuda_total_time_us": parse_microseconds(
            first_value(row, lookup, ("cuda_total_time_us", "cuda_time_total", "cuda_total_time", "cuda_time", "gpu_time_total", "gpu_time"))
        ),
        "source_file": source.name,
    }


def summarize_trace(inputs: Sequence[Path], output: Path, metadata_path: Path, top: int, run_name: str | None) -> dict[str, Any]:
    sources = expand_csv_inputs(inputs, output)
    aggregate: dict[tuple[str, str, str, str], dict[str, Any]] = {}
    input_rows = 0
    skipped_rows = 0
    for source in sources:
        fields, rows = read_csv(source)
        lookup = header_lookup(fields)
        for source_row in rows:
            input_rows += 1
            normalized = trace_row_from_source(source_row, lookup, source, run_name)
            if normalized is None:
                skipped_rows += 1
                continue
            key = (normalized["run_name"], normalized["stage"], normalized["op_name"], normalized["source_file"])
            current = aggregate.get(key)
            if current is None:
                aggregate[key] = normalized
                continue
            current["calls"] += normalized["calls"]
            for duration_field in ("cpu_self_time_us", "cpu_total_time_us", "cuda_self_time_us", "cuda_total_time_us"):
                current[duration_field] += normalized[duration_field]

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for normalized in aggregate.values():
        for field in ("cpu_self_time_us", "cpu_total_time_us", "cuda_self_time_us", "cuda_total_time_us"):
            normalized[field] = round(float(normalized[field]), 3)
        grouped[(normalized["run_name"], normalized["stage"])].append(normalized)

    output_rows: list[dict[str, Any]] = []
    for _, group_rows in sorted(grouped.items()):
        ordered = sorted(
            group_rows,
            key=lambda item: (item["cuda_total_time_us"], item["cpu_total_time_us"], item["calls"]),
            reverse=True,
        )
        output_rows.extend(ordered[:top])
    output_rows.sort(key=lambda item: (item["run_name"], item["stage"], -item["cuda_total_time_us"], item["op_name"]))
    atomic_write_csv(output, TRACE_COLUMNS, output_rows)

    metadata = {
        "schema_version": 1,
        "kind": "a2p_trace_summary",
        "generated_at_utc": utc_timestamp(),
        "source_files": [source.name for source in sources],
        "source_file_count": len(sources),
        "input_rows": input_rows,
        "skipped_rows_without_operation_name": skipped_rows,
        "rows_written": len(output_rows),
        "top_per_run_and_stage": top,
        "sort_order": "cuda_total_time_us descending, then cpu_total_time_us descending",
        "columns": {
            "stage": "显式 profiler/NVTX range；无显式 range 时仅按 op 名称保守推断。",
            "cpu_*_time_us": "累计 CPU 时间（微秒）。",
            "cuda_*_time_us": "累计 CUDA/GPU 时间（微秒）；CPU-only profile 可为 0。",
            "source_file": "仅原始汇总文件的 basename，避免提交本机目录。",
        },
        "raw_artifact_policy": "Chrome trace、.nsys-rep、SQLite 和完整 timeline 留在本地，不在该汇总中复制。",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def is_sensitive_column(column: str) -> bool:
    normalized = normalize_name(column)
    tokens = set(normalized.split("_"))
    return bool(tokens & SENSITIVE_COLUMN_TOKENS)


def memory_output_columns(input_fields: Sequence[str]) -> list[str]:
    safe_fields = [field for field in input_fields if not is_sensitive_column(field)]
    known = [field for field in MEMORY_PREFERRED_COLUMNS if field in safe_fields]
    unknown = [field for field in safe_fields if field not in known]
    return known + unknown


def summarize_memory(input_path: Path, output: Path, metadata_path: Path, include_failures: bool) -> dict[str, Any]:
    validate_input_path(input_path)
    if input_path.suffix.lower() != ".csv":
        raise ValueError("memory 汇总只接受 CSV peaks 表")
    fields, input_rows = read_csv(input_path)
    output_fields = memory_output_columns(fields)
    normalized_status_field = next((field for field in fields if normalize_name(field) == "status"), None)
    rows: list[dict[str, Any]] = []
    for source_row in input_rows:
        if not include_failures and normalized_status_field is not None:
            if source_row.get(normalized_status_field) not in {"completed", "completed_with_warnings"}:
                continue
        safe_row: dict[str, Any] = {}
        for field in output_fields:
            value = source_row.get(field, "")
            # Free-form diagnostics frequently contain source paths.  Keep
            # machine-readable numeric/config fields exact; redact only text.
            if normalize_name(field) in {"error_message", "warning_message", "fallback_reason"}:
                value = redact_text(value)
            safe_row[field] = value
        rows.append(safe_row)
    atomic_write_csv(output, output_fields, rows)

    statuses: dict[str, int] = defaultdict(int)
    if normalized_status_field is not None:
        for row in rows:
            statuses[str(row.get(normalized_status_field, ""))] += 1
    metadata = {
        "schema_version": 1,
        "kind": "a2p_memory_peaks_public_summary",
        "generated_at_utc": utc_timestamp(),
        "source_file": input_path.name,
        "input_rows": len(input_rows),
        "rows_written": len(rows),
        "include_failures": include_failures,
        "dropped_sensitive_columns": [field for field in fields if field not in output_fields],
        "status_counts": dict(sorted(statuses.items())),
        "raw_artifact_policy": "snapshot/Chrome trace 不会被读取或复制；CSV 中仅保留其 basename 字段。",
    }
    atomic_write_json(metadata_path, metadata)
    return metadata


def printable_command(args: argparse.Namespace) -> str:
    # Avoid echoing user-local directories.  This is informative console output
    # only, not public metadata.
    if args.command == "trace":
        return shlex.join(["python", "profiling/summarize.py", "trace", "--output", str(args.output), "--top", str(args.top)])
    return shlex.join(["python", "profiling/summarize.py", "memory", "--output", str(args.output)])


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        if args.command == "trace":
            metadata = summarize_trace(args.input, args.output, args.metadata, args.top, args.run_name)
            print(f"已写入 {metadata['rows_written']} 行 trace 汇总：{args.output}")
        else:
            metadata = summarize_memory(args.input, args.output, args.metadata, args.include_failures)
            print(f"已写入 {metadata['rows_written']} 行 memory peaks 汇总：{args.output}")
        print(f"metadata：{args.metadata}")
        return 0
    except (OSError, ValueError, csv.Error) as error:
        print(f"汇总失败：{redact_text(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
