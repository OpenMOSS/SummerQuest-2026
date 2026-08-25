#!/usr/bin/env python3
"""Build a small, redacted A2-P result package from local experiment outputs.

The profiler's Chrome traces, Nsight reports, memory snapshots, and pickles
remain local.  This program reads only the compact CSV/JSON entry points
produced by the profiling scripts and writes a reviewable public package.

Examples
--------

Prepare a new package without touching an existing destination::

    python profiling/prepare_submission.py \
        --source-root results --output-root results/public

Inspect the files and report aggregates that would be produced::

    python profiling/prepare_submission.py --dry-run

Replace a package created by an earlier run only after review::

    python profiling/prepare_submission.py --output-root results/public --overwrite

``--strict`` is useful immediately before copying the package to the public
submission repository: it requires every A2-P result category to be present.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import math
import os
import re
import shutil
import statistics
import sys
import tempfile
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Final, TypeAlias, cast


MEBIBYTE: Final = 1024**2
DEFAULT_MAX_INPUT_BYTES: Final = 16 * MEBIBYTE
DEFAULT_MAX_FILE_BYTES: Final = 5 * MEBIBYTE
DEFAULT_MAX_TOTAL_BYTES: Final = 2 * MEBIBYTE
MAX_SAFE_STRING_LENGTH: Final = 500
MAX_ARTIFACT_FILENAME_LENGTH: Final = 180
MAX_JSON_DEPTH: Final = 32
MAX_JSON_LIST_ITEMS: Final = 2_000
MAX_JSON_MAPPING_ITEMS: Final = 2_000

FORBIDDEN_SUFFIXES: Final[frozenset[str]] = frozenset(
    {".pickle", ".pkl", ".nsys-rep", ".sqlite", ".db", ".zip", ".tar", ".gz", ".pt", ".pth", ".ckpt"}
)
SENSITIVE_NAME_TOKENS: Final[frozenset[str]] = frozenset(
    {
        "account",
        "auth",
        "cookie",
        "credential",
        "directory",
        "email",
        "env",
        "environment_variable",
        "home",
        "host",
        "hostname",
        "ip",
        "key",
        "local_path",
        "password",
        "pid",
        "private",
        "process",
        "secret",
        "token",
        "snapshot",
        "trace",
        "trace_file",
        "user",
        "username",
        "uuid",
    }
)
ARTIFACT_FILENAME_KEYS: Final[frozenset[str]] = frozenset({"source_file"})

BENCHMARK_PREFERRED_COLUMNS: Final[tuple[str, ...]] = (
    "run_name",
    "timestamp_utc",
    "mode",
    "model_size",
    "batch_size",
    "context_length",
    "vocab_size",
    "dtype",
    "device",
    "warmup_steps",
    "measurement_steps",
    "seed",
    "parameter_count",
    "measurement_index",
    "elapsed_ms",
    "mean_ms",
    "std_ms",
    "cv_percent",
)
TRACE_COLUMNS: Final[tuple[str, ...]] = (
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
MEMORY_PREFERRED_COLUMNS: Final[tuple[str, ...]] = (
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

JsonScalar: TypeAlias = str | int | float | bool | None
JsonValue: TypeAlias = JsonScalar | list["JsonValue"] | dict[str, "JsonValue"]
CsvCell: TypeAlias = str | int | float | None

IPV4_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b"
)
UNIX_PATH_PATTERN: Final[re.Pattern[str]] = re.compile(r"(?<![\w<])/(?:[^\s:'\"()\[\],]+/)*[^\s:'\"()\[\],]+")
WINDOWS_PATH_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[A-Za-z]:\\(?:[^\\\s:'\"()\[\],]+\\)*[^\\\s:'\"()\[\],]+")
IDENTITY_PATTERN: Final[re.Pattern[str]] = re.compile(r"\b[^\s@/:]+@[^\s@/:]+\b")
NUMBER_PATTERN: Final[re.Pattern[str]] = re.compile(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?", re.IGNORECASE)
NUMBER_WITH_UNIT_PATTERN: Final[re.Pattern[str]] = re.compile(
    r"([+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(ns|us|µs|ms|s)?", re.IGNORECASE
)

_OMITTED: Final[object] = object()


class SubmissionPreparationError(ValueError):
    """Raised when a local artifact is unsafe or cannot form a public package."""


@dataclass(slots=True)
class BenchmarkData:
    """Normalized benchmark CSV rows and their non-identifying provenance."""

    fields: list[str]
    rows: list[dict[str, str]]
    source_count: int
    input_rows: int


@dataclass(slots=True)
class BenchmarkInputs:
    """Either a directory discovery source or an explicit list of benchmark CSVs."""

    directory: Path
    files: tuple[Path, ...] | None


@dataclass(slots=True)
class ProfileData:
    """Public-safe trace rows and per-run metadata documents."""

    trace_rows: list[dict[str, CsvCell]]
    trace_source_count: int
    input_rows: int
    skipped_rows: int
    metadata_documents: list[dict[str, JsonValue]]


@dataclass(slots=True)
class MemoryData:
    """Sanitized memory peaks and optional metadata."""

    fields: list[str]
    rows: list[dict[str, str]]
    metadata: dict[str, JsonValue] | None


@dataclass(slots=True)
class MixedPrecisionData:
    """One or more merged, sanitized mixed-precision result documents."""

    document: dict[str, JsonValue] | None


@dataclass(frozen=True, slots=True)
class Artifact:
    """One generated public file, kept in memory until validation succeeds."""

    relative_path: Path
    content: bytes


@dataclass(frozen=True, slots=True)
class SourcePaths:
    """Local-only input locations; none are written into public output."""

    benchmark: BenchmarkInputs
    profile_dir: Path
    memory_dir: Path
    mixed_inputs: tuple[Path, ...]


def utc_timestamp() -> str:
    """Return an ISO-8601 UTC timestamp for generated public metadata."""

    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def normalize_name(value: str) -> str:
    """Convert an arbitrary CSV/JSON key to a stable lowercase identifier."""

    return re.sub(r"[^a-z0-9]+", "_", value.strip().lower()).strip("_")


def redact_text(value: object, limit: int = MAX_SAFE_STRING_LENGTH) -> str:
    """Remove common local identifiers while preserving useful diagnostics."""

    text = " ".join(str(value).split())
    text = IPV4_PATTERN.sub("<ip>", text)
    text = UUID_PATTERN.sub("<id>", text)
    text = IDENTITY_PATTERN.sub("<identity>", text)
    text = WINDOWS_PATH_PATTERN.sub("<path>", text)
    text = UNIX_PATH_PATTERN.sub("<path>", text)
    return text[:limit]


def redact_oom_diagnostic(value: object) -> str:
    """Keep an OOM diagnosis useful without exposing allocator/process details."""

    text = redact_text(value, 2_000)
    allocation = re.search(r"Tried to allocate\s+([0-9.]+\s+(?:KiB|MiB|GiB))", text, re.IGNORECASE)
    if allocation is not None:
        return f"CUDA out of memory; failed allocation request: {allocation.group(1)}."
    return "CUDA out of memory during the recorded workload."


def safe_artifact_filename(value: object) -> str:
    """Keep only a bounded basename when metadata refers to a local artifact."""

    candidate = str(value).replace("\\", "/").rsplit("/", maxsplit=1)[-1]
    candidate = redact_text(candidate, MAX_ARTIFACT_FILENAME_LENGTH)
    return "" if candidate in {"", ".", ".."} else candidate


def is_sensitive_name(name: str | None) -> bool:
    """Return whether a field/key would directly disclose local identity data."""

    if not name:
        return False
    normalized = normalize_name(name)
    tokens = set(normalized.split("_"))
    if tokens & SENSITIVE_NAME_TOKENS:
        return True
    return normalized in {"location", "local", "local_file", "local_directory"}


def sanitize_json(value: object, *, key: str | None = None, depth: int = 0) -> JsonValue | object:
    """Recursively redact JSON while bounding deeply nested or huge documents."""

    if is_sensitive_name(key):
        return _OMITTED
    if depth > MAX_JSON_DEPTH:
        return "<truncated>"
    if isinstance(value, Mapping):
        sanitized: dict[str, JsonValue] = {}
        for index, (raw_key, raw_value) in enumerate(value.items()):
            if index >= MAX_JSON_MAPPING_ITEMS:
                sanitized["truncated_items"] = True
                break
            if not isinstance(raw_key, str):
                continue
            normalized_key = normalize_name(raw_key)
            if normalized_key == "privacy" and isinstance(raw_value, Mapping):
                sanitized[normalized_key] = {"omitted_sensitive_fields": True}
                continue
            if normalized_key == "trace_file":
                basename = safe_artifact_filename(raw_value)
                if basename:
                    sanitized["local_trace_basename"] = basename
                continue
            cleaned = sanitize_json(raw_value, key=raw_key, depth=depth + 1)
            if cleaned is not _OMITTED:
                sanitized[normalized_key or "unnamed"] = cast(JsonValue, cleaned)
        return sanitized
    if isinstance(value, list):
        sanitized_items: list[JsonValue] = []
        for item in value[:MAX_JSON_LIST_ITEMS]:
            cleaned = sanitize_json(item, key=key, depth=depth + 1)
            if cleaned is not _OMITTED:
                sanitized_items.append(cast(JsonValue, cleaned))
        return sanitized_items
    if isinstance(value, bool) or value is None:
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, str):
        if normalize_name(key or "") == "error_message" and "out of memory" in value.lower():
            return redact_oom_diagnostic(value)
        return redact_text(value)
    return redact_text(value)


def positive_int(value: str) -> int:
    """Parse a strictly positive command-line integer."""

    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("必须是正整数") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def relative_output_path(value: str) -> Path:
    """Accept an output-relative path and reject paths that can escape the root."""

    path = Path(value)
    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise argparse.ArgumentTypeError("输出文件必须是 output root 内的相对路径")
    return path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    """Build the CLI without requiring PyTorch or any third-party package."""

    parser = argparse.ArgumentParser(
        description="将本地 A2-P 轻量结果整理为可公开提交的数据包；不会复制 trace、snapshot 或 pickle。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--source-root", type=Path, default=Path("results"), help="本地实验结果根目录")
    parser.add_argument("--benchmark-dir", type=Path, default=None, help="默认 <source-root>/benchmark")
    parser.add_argument(
        "--benchmark-input",
        type=Path,
        action="append",
        default=None,
        help="显式选择一个或多个正式 benchmark CSV；重复传入以排除 smoke 文件",
    )
    parser.add_argument("--profile-dir", type=Path, default=None, help="默认 <source-root>/profile")
    parser.add_argument("--memory-dir", type=Path, default=None, help="默认 <source-root>/memory")
    parser.add_argument(
        "--mixed-input",
        type=Path,
        action="append",
        default=None,
        help="一个或多个 mixed precision JSON；重复传入时按 mode 合并",
    )
    parser.add_argument("--output-root", type=Path, default=Path("results") / "public", help="公开结果输出目录")
    parser.add_argument("--overwrite", action="store_true", help="显式替换已有 output root；默认拒绝覆写")
    parser.add_argument("--strict", action="store_true", help="要求 benchmark/profile/mixed/memory 的全部必交汇总均存在")
    parser.add_argument("--dry-run", action="store_true", help="只校验、汇总并打印计划，不写入 output root")
    parser.add_argument(
        "--report-data",
        type=relative_output_path,
        default=Path("report_data.json"),
        help="在 output root 内输出面向 README 的统计 JSON",
    )
    parser.add_argument("--no-report-data", dest="report_data", action="store_const", const=None, help="不输出 report_data.json")
    parser.add_argument(
        "--max-profile-rows-per-run-stage",
        type=positive_int,
        default=100,
        help="每个 run/stage 保留的累计时间最高操作数；用于控制公开附件体积",
    )
    parser.add_argument("--max-input-bytes", type=positive_int, default=DEFAULT_MAX_INPUT_BYTES, help="单个允许读取的轻量输入上限")
    parser.add_argument("--max-file-bytes", type=positive_int, default=DEFAULT_MAX_FILE_BYTES, help="单个生成文件上限")
    parser.add_argument("--max-total-bytes", type=positive_int, default=DEFAULT_MAX_TOTAL_BYTES, help="生成 results 附件总大小上限")
    return parser.parse_args(argv)


def ensure_allowed_input_file(path: Path, max_input_bytes: int) -> None:
    """Reject unavailable, oversized, or explicitly forbidden raw artifacts."""

    if not path.is_file():
        raise SubmissionPreparationError(f"找不到输入文件：{path.name}")
    if path.suffix.lower() in FORBIDDEN_SUFFIXES:
        raise SubmissionPreparationError(f"拒绝读取禁止公开的原始 artifact：{path.name}")
    if path.stat().st_size > max_input_bytes:
        raise SubmissionPreparationError(f"输入文件超过轻量处理上限：{path.name}")


def read_csv(path: Path, max_input_bytes: int) -> tuple[list[str], list[dict[str, str]]]:
    """Read a bounded UTF-8 CSV without retaining malformed extra columns."""

    ensure_allowed_input_file(path, max_input_bytes)
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise SubmissionPreparationError(f"CSV 缺少 header：{path.name}")
        fields = [str(field) for field in reader.fieldnames if field is not None]
        if not fields:
            raise SubmissionPreparationError(f"CSV 缺少有效 header：{path.name}")
        rows: list[dict[str, str]] = []
        for raw_row in reader:
            row: dict[str, str] = {}
            for key, value in raw_row.items():
                if key is not None:
                    row[str(key)] = "" if value is None else str(value)
            rows.append(row)
    return fields, rows


def load_json_document(path: Path, max_input_bytes: int, *, expected_mixed_precision: bool = False) -> dict[str, JsonValue]:
    """Load JSON only when it is a compact metadata/result document, never a trace."""

    ensure_allowed_input_file(path, max_input_bytes)
    with path.open("r", encoding="utf-8") as handle:
        raw_document = json.load(handle)
    if not isinstance(raw_document, Mapping):
        raise SubmissionPreparationError(f"JSON 根节点必须是对象：{path.name}")
    raw_keys = {str(key) for key in raw_document}
    if {"traceEvents", "systemTraceEvents", "displayTimeUnit"} & raw_keys:
        raise SubmissionPreparationError(f"拒绝读取 profiler Chrome trace：{path.name}")
    if expected_mixed_precision and not ({"accumulation", "language_model_benchmark", "status"} & raw_keys):
        raise SubmissionPreparationError(f"mixed precision JSON 不含预期实验字段：{path.name}")
    cleaned = sanitize_json(raw_document)
    if not isinstance(cleaned, dict):
        raise SubmissionPreparationError(f"无法清洗 JSON：{path.name}")
    return cast(dict[str, JsonValue], cleaned)


def safe_header_mapping(fields: Sequence[str]) -> dict[str, str]:
    """Map original fields to safe normalized names, dropping sensitive collisions."""

    mapping: dict[str, str] = {}
    claimed: set[str] = set()
    for field in fields:
        normalized = normalize_name(field)
        if not normalized or is_sensitive_name(normalized) or normalized in claimed:
            continue
        mapping[field] = normalized
        claimed.add(normalized)
    return mapping


def order_fields(fields: Iterable[str], preferred: Sequence[str]) -> list[str]:
    """Place known schema columns first and make unknown fields deterministic."""

    field_set = set(fields)
    ordered = [field for field in preferred if field in field_set]
    ordered.extend(sorted(field for field in field_set if field not in ordered))
    return ordered


def clean_csv_value(field: str, value: object) -> str:
    """Redact a CSV cell after sensitive artifact-reference headers are dropped."""

    return redact_text(value)


def clean_csv_row(raw_row: Mapping[str, str], header_mapping: Mapping[str, str]) -> dict[str, str]:
    """Apply the safe header/value policy to one CSV row."""

    cleaned: dict[str, str] = {}
    for original_name, safe_name in header_mapping.items():
        value = raw_row.get(original_name, "")
        cleaned[safe_name] = (
            redact_oom_diagnostic(value)
            if safe_name == "error_message" and "out of memory" in value.lower()
            else clean_csv_value(safe_name, value)
        )
    return cleaned


def lookup_headers(fields: Sequence[str]) -> dict[str, str]:
    """Return a normalized-name to original-header lookup for source CSVs."""

    lookup: dict[str, str] = {}
    for field in fields:
        lookup.setdefault(normalize_name(field), field)
    return lookup


def first_csv_value(row: Mapping[str, str], lookup: Mapping[str, str], aliases: Sequence[str], default: str = "") -> str:
    """Read the first non-empty matching alias from a source CSV row."""

    for alias in aliases:
        field = lookup.get(normalize_name(alias))
        if field is None:
            continue
        value = row.get(field, "").strip()
        if value:
            return value
    return default


def parse_finite_number(value: object) -> float | None:
    """Parse a plain finite number without treating booleans as measurements."""

    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        number = float(value)
    else:
        text = str(value).strip().replace(",", "")
        if not NUMBER_PATTERN.fullmatch(text):
            return None
        number = float(text)
    return number if math.isfinite(number) else None


def parse_microseconds(value: object) -> float:
    """Parse profiler timing strings into microseconds; malformed values become zero."""

    compact = str(value).strip().replace(",", "")
    match = NUMBER_WITH_UNIT_PATTERN.fullmatch(compact)
    if match is None:
        return 0.0
    number = parse_finite_number(match.group(1))
    if number is None:
        return 0.0
    unit = (match.group(2) or "us").lower()
    multiplier = {"ns": 0.001, "us": 1.0, "µs": 1.0, "ms": 1_000.0, "s": 1_000_000.0}[unit]
    return number * multiplier


def parse_milliseconds(value: object) -> float | None:
    """Parse a benchmark timing value into milliseconds."""

    compact = str(value).strip().replace(",", "")
    match = NUMBER_WITH_UNIT_PATTERN.fullmatch(compact)
    if match is None:
        return None
    number = parse_finite_number(match.group(1))
    if number is None:
        return None
    unit = (match.group(2) or "ms").lower()
    multiplier = {"ns": 0.000001, "us": 0.001, "µs": 0.001, "ms": 1.0, "s": 1_000.0}[unit]
    return number * multiplier


def parse_count(value: object) -> int:
    """Parse non-negative call counts conservatively."""

    number = parse_finite_number(value)
    if number is None or number <= 0:
        return 0
    return int(number)


def infer_stage(explicit_stage: str, operation: str) -> str:
    """Use explicit profiler ranges first and only then conservative name inference."""

    if explicit_stage:
        return explicit_stage
    name = operation.lower()
    if "profile/warmup" in name or "profile_warmup" in name or "warmup" in name:
        return "profile/warmup"
    if "profile/measure" in name or "profile_measure" in name or "measure" in name:
        return "profile/measure"
    if "attention/scores" in name or "attention_scores" in name or ("attention" in name and "score" in name):
        return "attention/scores"
    if "attention/softmax" in name or "attention_softmax" in name or "softmax" in name:
        return "attention/softmax"
    if "attention/value" in name or "attention_value" in name or ("attention" in name and "value" in name):
        return "attention/value"
    if "backward" in name or "autograd" in name:
        return "backward"
    if "optimizer" in name or "adam" in name or "zero_grad" in name:
        return "optimizer"
    if "forward" in name:
        return "forward"
    return "unlabelled"


def collect_benchmark(inputs: BenchmarkInputs, max_input_bytes: int) -> BenchmarkData:
    """Merge selected benchmark CSVs while retaining raw timing rows."""

    benchmark_dir = inputs.directory
    if inputs.files is None:
        if not benchmark_dir.exists():
            return BenchmarkData([], [], 0, 0)
        if not benchmark_dir.is_dir():
            raise SubmissionPreparationError("benchmark 输入必须是目录")
        source_files = sorted(path for path in benchmark_dir.glob("*.csv") if path.is_file())
    else:
        source_files = sorted(dict.fromkeys(inputs.files))
        for source in source_files:
            ensure_allowed_input_file(source, max_input_bytes)
            if source.suffix.lower() != ".csv":
                raise SubmissionPreparationError("--benchmark-input 只接受 CSV 文件")
    all_rows: list[dict[str, str]] = []
    all_fields: set[str] = set()
    input_rows = 0
    for ordinal, source in enumerate(source_files, start=1):
        fields, source_rows = read_csv(source, max_input_bytes)
        mapping = safe_header_mapping(fields)
        if not mapping:
            raise SubmissionPreparationError(f"benchmark CSV 没有可公开的列：{source.name}")
        source_label = f"benchmark_{ordinal:03d}.csv"
        for raw_row in source_rows:
            cleaned = clean_csv_row(raw_row, mapping)
            cleaned["submission_source"] = source_label
            all_rows.append(cleaned)
        all_fields.update(mapping.values())
        input_rows += len(source_rows)

    if source_files:
        all_fields.add("submission_source")
    return BenchmarkData(order_fields(all_fields, (*BENCHMARK_PREFERRED_COLUMNS, "submission_source")), all_rows, len(source_files), input_rows)


def trace_row_from_source(
    row: Mapping[str, str],
    lookup: Mapping[str, str],
    *,
    source_label: str,
) -> dict[str, CsvCell] | None:
    """Normalize one trace-summary row into the fixed public CSV schema."""

    operation = first_csv_value(row, lookup, ("op_name", "operator", "op", "kernel_name", "kernel", "name", "key"))
    if not operation:
        return None
    run_name = first_csv_value(row, lookup, ("run_name", "run", "profile_name", "trace_name"), source_label.removesuffix(".csv"))
    explicit_stage = first_csv_value(row, lookup, ("stage", "stage_range", "range", "nvtx_range"))
    return {
        "run_name": redact_text(run_name, 120),
        "stage": redact_text(infer_stage(explicit_stage, operation), 120),
        "op_name": redact_text(operation, 300),
        "calls": parse_count(first_csv_value(row, lookup, ("calls", "count", "number_of_calls"))),
        "cpu_self_time_us": parse_microseconds(
            first_csv_value(row, lookup, ("cpu_self_time_us", "self_cpu_time_total", "self_cpu_time_us", "cpu_self_time"))
        ),
        "cpu_total_time_us": parse_microseconds(
            first_csv_value(row, lookup, ("cpu_total_time_us", "cpu_time_total", "cpu_total_time", "cpu_time"))
        ),
        "cuda_self_time_us": parse_microseconds(
            first_csv_value(
                row,
                lookup,
                ("cuda_self_time_us", "self_cuda_time_total", "self_device_time_total", "self_cuda_time", "self_device_time"),
            )
        ),
        "cuda_total_time_us": parse_microseconds(
            first_csv_value(
                row,
                lookup,
                ("cuda_total_time_us", "cuda_time_total", "device_time_total", "cuda_time", "gpu_time_total", "gpu_time"),
            )
        ),
        "source_file": source_label,
    }


def collect_profile(profile_dir: Path, max_input_bytes: int, max_rows_per_group: int) -> ProfileData:
    """Collect only compact trace summaries and explicitly named run metadata."""

    if not profile_dir.exists():
        return ProfileData([], 0, 0, 0, [])
    if not profile_dir.is_dir():
        raise SubmissionPreparationError("profile 输入必须是目录")

    trace_files = sorted(path for path in profile_dir.rglob("trace_summary.csv") if path.is_file())
    input_rows = 0
    skipped_rows = 0
    normalized_rows: list[dict[str, CsvCell]] = []
    for ordinal, source in enumerate(trace_files, start=1):
        fields, rows = read_csv(source, max_input_bytes)
        lookup = lookup_headers(fields)
        source_label = f"profile_{ordinal:03d}.csv"
        for row in rows:
            input_rows += 1
            normalized = trace_row_from_source(row, lookup, source_label=source_label)
            if normalized is None:
                skipped_rows += 1
                continue
            # CPU record_function rows and GPU annotation rows can share a stage
            # name but describe asynchronous work. Keep both rows rather than
            # summing them into an invalid synthetic range.
            for timing_field in ("cpu_self_time_us", "cpu_total_time_us", "cuda_self_time_us", "cuda_total_time_us"):
                normalized[timing_field] = round(float(normalized[timing_field] or 0.0), 3)
            normalized_rows.append(normalized)

    grouped: dict[tuple[str, str], list[dict[str, CsvCell]]] = defaultdict(list)
    for row in normalized_rows:
        grouped[(str(row["run_name"]), str(row["stage"]))].append(row)

    trace_rows: list[dict[str, CsvCell]] = []
    for group_rows in grouped.values():
        ordered = sorted(
            group_rows,
            key=lambda item: (
                float(item["cuda_total_time_us"] or 0.0),
                float(item["cpu_total_time_us"] or 0.0),
                parse_count(item["calls"]),
            ),
            reverse=True,
        )
        trace_rows.extend(ordered[:max_rows_per_group])
    trace_rows.sort(
        key=lambda item: (
            str(item["run_name"]),
            str(item["stage"]),
            -float(item["cuda_total_time_us"] or 0.0),
            str(item["op_name"]),
        )
    )

    metadata_documents: list[dict[str, JsonValue]] = []
    metadata_files = sorted(path for path in profile_dir.rglob("run_metadata.json") if path.is_file())
    for ordinal, source in enumerate(metadata_files, start=1):
        document = load_json_document(source, max_input_bytes)
        document["submission_source_label"] = f"profile_metadata_{ordinal:03d}.json"
        metadata_documents.append(document)
    return ProfileData(trace_rows, len(trace_files), input_rows, skipped_rows, metadata_documents)


def collect_memory(memory_dir: Path, max_input_bytes: int) -> MemoryData:
    """Read only ``peaks.csv`` and ``run_metadata.json`` from memory results."""

    if not memory_dir.exists():
        return MemoryData([], [], None)
    if not memory_dir.is_dir():
        raise SubmissionPreparationError("memory 输入必须是目录")

    peaks_path = memory_dir / "peaks.csv"
    rows: list[dict[str, str]] = []
    fields: list[str] = []
    if peaks_path.is_file():
        source_fields, source_rows = read_csv(peaks_path, max_input_bytes)
        mapping = safe_header_mapping(source_fields)
        if not mapping:
            raise SubmissionPreparationError("memory peaks CSV 没有可公开的列")
        rows = [clean_csv_row(row, mapping) for row in source_rows]
        fields = order_fields(mapping.values(), MEMORY_PREFERRED_COLUMNS)

    metadata_path = memory_dir / "run_metadata.json"
    metadata = load_json_document(metadata_path, max_input_bytes) if metadata_path.is_file() else None
    return MemoryData(fields, rows, metadata)


def collect_mixed_precision(input_paths: Sequence[Path], max_input_bytes: int) -> MixedPrecisionData:
    """Merge one or more compact mixed-precision documents by benchmark mode."""

    existing_paths = [path for path in input_paths if path.exists()]
    if not existing_paths:
        return MixedPrecisionData(None)

    documents = [load_json_document(path, max_input_bytes, expected_mixed_precision=True) for path in existing_paths]
    merged = dict(documents[0])
    merged_modes: dict[str, JsonValue] = {}
    source_labels: list[str] = []
    for ordinal, document in enumerate(documents, start=1):
        language_model = json_mapping(document.get("language_model_benchmark"))
        modes = json_mapping(language_model.get("modes"))
        for mode_name, mode_payload in modes.items():
            if mode_name in merged_modes and merged_modes[mode_name] != mode_payload:
                raise SubmissionPreparationError(f"mixed precision mode 重复且内容不一致：{mode_name}")
            merged_modes[mode_name] = mode_payload
        source_labels.append(f"mixed_precision_{ordinal:03d}.json")

    language_model = json_mapping(merged.get("language_model_benchmark"))
    language_model["modes"] = dict(sorted(merged_modes.items()))
    experiment_config = json_mapping(language_model.get("experiment_config"))
    if len(merged_modes) > 1:
        experiment_config["mode"] = "multiple"
    language_model["experiment_config"] = experiment_config
    merged["language_model_benchmark"] = language_model
    merged["submission_source_labels"] = source_labels
    return MixedPrecisionData(merged)


def row_value(row: Mapping[str, str], *aliases: str) -> str:
    """Read a normalized public CSV value through a list of aliases."""

    for alias in aliases:
        value = row.get(normalize_name(alias), "")
        if value:
            return value
    return ""


def optional_number(value: object) -> float | None:
    """Return a finite number or ``None`` for a JSON report field."""

    return parse_finite_number(value)


def rounded_number(value: float | None, digits: int = 6) -> float | None:
    """Round finite report data while retaining ``None`` for unavailable metrics."""

    return None if value is None else round(value, digits)


def benchmark_report(data: BenchmarkData) -> dict[str, JsonValue]:
    """Derive benchmark run statistics from merged raw timing rows."""

    if not data.rows:
        return {
            "status": "missing",
            "source_file_count": data.source_count,
            "input_rows": data.input_rows,
            "runs": [],
        }

    grouped: dict[tuple[str, ...], list[dict[str, str]]] = defaultdict(list)
    for row in data.rows:
        key = tuple(
            row_value(
                row,
                alias,
            )
            for alias in (
                "submission_source",
                "run_name",
                "timestamp_utc",
                "mode",
                "model_size",
                "batch_size",
                "context_length",
                "dtype",
                "warmup_steps",
                "seed",
            )
        )
        grouped[key].append(row)

    runs: list[dict[str, JsonValue]] = []
    for _, rows in sorted(grouped.items()):
        first = rows[0]
        timings = [timing for row in rows if (timing := parse_milliseconds(row_value(row, "elapsed_ms"))) is not None]
        mean_ms = statistics.fmean(timings) if timings else optional_number(row_value(first, "mean_ms"))
        std_ms = statistics.stdev(timings) if len(timings) > 1 else optional_number(row_value(first, "std_ms", "sample_std_ms"))
        cv_percent = (
            100.0 * std_ms / mean_ms
            if mean_ms is not None and std_ms is not None and mean_ms > 0.0
            else optional_number(row_value(first, "cv_percent"))
        )
        run: dict[str, JsonValue] = {
            "run_name": row_value(first, "run_name") or "unnamed",
            "mode": row_value(first, "mode") or None,
            "model_size": row_value(first, "model_size") or None,
            "batch_size": optional_number(row_value(first, "batch_size")),
            "context_length": optional_number(row_value(first, "context_length")),
            "dtype": row_value(first, "dtype") or None,
            "warmup_steps": optional_number(row_value(first, "warmup_steps")),
            "measurement_count": len(timings) if timings else len(rows),
            "raw_timings_ms": [round(value, 6) for value in timings[:1_000]],
            "mean_ms": rounded_number(mean_ms),
            "sample_std_ms": rounded_number(std_ms),
            "cv_percent": rounded_number(cv_percent),
        }
        timestamp = row_value(first, "timestamp_utc")
        if timestamp:
            run["timestamp_utc"] = timestamp
        parameter_count = optional_number(row_value(first, "parameter_count"))
        if parameter_count is not None:
            run["parameter_count"] = int(parameter_count)
        runs.append(run)

    return {
        "status": "available",
        "source_file_count": data.source_count,
        "input_rows": data.input_rows,
        "rows_written": len(data.rows),
        "runs": runs,
    }


def profile_report(data: ProfileData) -> dict[str, JsonValue]:
    """Summarize explicit stage ranges without falsely summing nested ops."""

    if not data.trace_rows:
        return {
            "status": "missing",
            "trace_source_file_count": data.trace_source_count,
            "input_rows": data.input_rows,
            "skipped_rows": data.skipped_rows,
            "metadata_run_count": len(data.metadata_documents),
            "stage_summaries": [],
        }

    grouped: dict[tuple[str, str], list[dict[str, CsvCell]]] = defaultdict(list)
    for row in data.trace_rows:
        grouped[(str(row["run_name"]), str(row["stage"]))].append(row)

    stage_summaries: list[dict[str, JsonValue]] = []
    for (run_name, stage), rows in sorted(grouped.items()):
        normalized_stage = normalize_name(stage)
        direct_ranges = [row for row in rows if normalize_name(str(row["op_name"])) == normalized_stage]
        non_range_rows = [row for row in rows if row not in direct_ranges]
        top_pool = non_range_rows or rows
        top_row = max(
            top_pool,
            key=lambda row: (float(row["cuda_total_time_us"] or 0.0), float(row["cpu_total_time_us"] or 0.0)),
        )
        if direct_ranges:
            cpu_total = sum(float(row["cpu_total_time_us"] or 0.0) for row in direct_ranges)
            cuda_total = sum(float(row["cuda_total_time_us"] or 0.0) for row in direct_ranges)
            calls = sum(parse_count(row["calls"]) for row in direct_ranges)
            basis = "explicit_stage_range"
        else:
            cpu_total = None
            cuda_total = None
            calls = 0
            basis = "no_explicit_stage_range"
        stage_summaries.append(
            {
                "run_name": run_name,
                "stage": stage,
                "aggregation_basis": basis,
                "explicit_range_calls": calls,
                "explicit_range_cpu_total_time_us": rounded_number(cpu_total, 3),
                "explicit_range_cuda_total_time_us": rounded_number(cuda_total, 3),
                "top_operation": {
                    "op_name": str(top_row["op_name"]),
                    "calls": parse_count(top_row["calls"]),
                    "cpu_total_time_us": rounded_number(float(top_row["cpu_total_time_us"] or 0.0), 3),
                    "cuda_total_time_us": rounded_number(float(top_row["cuda_total_time_us"] or 0.0), 3),
                },
            }
        )
    return {
        "status": "available",
        "trace_source_file_count": data.trace_source_count,
        "input_rows": data.input_rows,
        "skipped_rows": data.skipped_rows,
        "rows_written": len(data.trace_rows),
        "metadata_run_count": len(data.metadata_documents),
        "stage_summaries": stage_summaries,
    }


def json_mapping(value: JsonValue | None) -> dict[str, JsonValue]:
    """Return a JSON object as a typed mapping, or an empty mapping otherwise."""

    return value if isinstance(value, dict) else {}


def json_list(value: JsonValue | None) -> list[JsonValue]:
    """Return a JSON array as a typed list, or an empty list otherwise."""

    return value if isinstance(value, list) else []


def mixed_precision_report(data: MixedPrecisionData) -> dict[str, JsonValue]:
    """Extract compact report-ready facts from the sanitized experiment JSON."""

    if data.document is None:
        return {"status": "missing", "accumulation_cases": [], "benchmark_modes": []}

    document = data.document
    accumulation = json_mapping(document.get("accumulation"))
    cases: list[dict[str, JsonValue]] = []
    for raw_case in json_list(accumulation.get("cases")):
        case = json_mapping(raw_case)
        if not case:
            continue
        cases.append(
            {
                "case": cast(JsonValue, case.get("case")),
                "result": optional_number(case.get("result")),
                "expected_mathematical_sum": optional_number(case.get("expected_mathematical_sum")),
                "absolute_error": optional_number(case.get("absolute_error")),
            }
        )

    toy_probe = json_mapping(document.get("toy_model_dtype_probe"))
    toy_summary = {
        field: cast(JsonValue, toy_probe.get(field))
        for field in (
            "autocast_dtype",
            "parameter_dtypes",
            "fc1_output_dtype",
            "layer_norm_output_dtype",
            "logits_dtype",
            "loss_dtype",
            "gradient_dtypes",
            "loss_is_finite",
        )
        if field in toy_probe
    }

    language_model = json_mapping(document.get("language_model_benchmark"))
    modes = json_mapping(language_model.get("modes"))
    mode_summaries: list[dict[str, JsonValue]] = []
    for mode_name, raw_mode in sorted(modes.items()):
        mode = json_mapping(raw_mode)
        fp32 = json_mapping(mode.get("fp32"))
        bf16 = json_mapping(mode.get("bf16_autocast"))
        comparison = json_mapping(mode.get("comparison"))
        fp32_timing = json_mapping(fp32.get("timing"))
        bf16_timing = json_mapping(bf16.get("timing"))
        fp32_memory = json_mapping(json_mapping(fp32.get("memory")).get("peak"))
        bf16_memory = json_mapping(json_mapping(bf16.get("memory")).get("peak"))
        mode_summaries.append(
            {
                "mode": mode_name,
                "fp32_mean_ms": rounded_number(optional_number(fp32_timing.get("mean_ms"))),
                "bf16_autocast_mean_ms": rounded_number(optional_number(bf16_timing.get("mean_ms"))),
                "bf16_over_fp32_speedup": rounded_number(optional_number(comparison.get("bf16_over_fp32_speedup"))),
                "fp32_peak_allocated_bytes": optional_number(fp32_memory.get("peak_allocated_bytes")),
                "bf16_peak_allocated_bytes": optional_number(bf16_memory.get("peak_allocated_bytes")),
                "peak_allocated_memory_delta_bytes_bf16_minus_fp32": optional_number(
                    comparison.get("peak_allocated_memory_delta_bytes_bf16_minus_fp32")
                ),
            }
        )

    return {
        "status": cast(JsonValue, document.get("status", "available")),
        "accumulation_cases": cases,
        "toy_model_dtype_probe": toy_summary,
        "language_model_config": json_mapping(language_model.get("model_config")),
        "experiment_config": json_mapping(language_model.get("experiment_config")),
        "benchmark_modes": mode_summaries,
    }


def memory_report(data: MemoryData) -> dict[str, JsonValue]:
    """Expose memory peak rows and maxima without adding raw snapshot content."""

    if not data.rows:
        return {"status": "missing", "rows_written": 0, "status_counts": {}, "peaks": []}

    statuses: Counter[str] = Counter(row_value(row, "status") or "unknown" for row in data.rows)
    selected_fields = (
        "run_id",
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
        "fallback_level",
        "fallback_reason",
        "active_mib_peak",
        "allocated_mib_peak",
        "reserved_mib_peak",
        "peak_allocated_mib",
        "peak_reserved_mib",
        "failure_stage",
        "exception_type",
    )
    numeric_fields = {
        "requested_context_length",
        "requested_batch_size",
        "context_length",
        "batch_size",
        "fallback_level",
        "active_mib_peak",
        "allocated_mib_peak",
        "reserved_mib_peak",
        "peak_allocated_mib",
        "peak_reserved_mib",
    }
    peaks: list[dict[str, JsonValue]] = []
    for row in data.rows:
        peak: dict[str, JsonValue] = {}
        for field in selected_fields:
            value = row_value(row, field)
            if not value:
                continue
            peak[field] = rounded_number(optional_number(value), 3) if field in numeric_fields else value
        peaks.append(peak)

    def maximum(field: str) -> float | None:
        values = [optional_number(row_value(row, field)) for row in data.rows]
        finite_values = [value for value in values if value is not None]
        return rounded_number(max(finite_values), 3) if finite_values else None

    return {
        "status": "available",
        "rows_written": len(data.rows),
        "status_counts": dict(sorted(statuses.items())),
        "max_active_mib_peak": maximum("active_mib_peak"),
        "max_peak_allocated_mib": maximum("peak_allocated_mib"),
        "max_peak_reserved_mib": maximum("peak_reserved_mib"),
        "peaks": peaks,
    }


def csv_bytes(fieldnames: Sequence[str], rows: Iterable[Mapping[str, CsvCell | str]]) -> bytes:
    """Serialize CSV deterministically with UTF-8 and a final newline."""

    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=list(fieldnames), extrasaction="ignore")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue().encode("utf-8")


def json_bytes(payload: Mapping[str, JsonValue]) -> bytes:
    """Serialize strict, readable JSON for a public package."""

    return (json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode("utf-8")


def profile_metadata_payload(data: ProfileData) -> dict[str, JsonValue]:
    """Flatten public profiler metadata while retaining only safe trace basenames."""

    flattened_runs: list[dict[str, JsonValue]] = []
    for document in data.metadata_documents:
        raw_runs = document.get("runs")
        if isinstance(raw_runs, list):
            candidates = raw_runs
        else:
            candidates = [document]
        for raw_run in candidates:
            if not isinstance(raw_run, dict):
                continue
            run = dict(raw_run)
            trace_file = run.pop("trace_file", None)
            if trace_file:
                run["local_trace_basename"] = safe_artifact_filename(trace_file)
            command = run.get("command")
            if isinstance(command, str):
                run["command"] = command.replace("--trace-dir results/profile/raw_v2", "--trace-dir results/profile")
            flattened_runs.append(run)
    flattened_runs.sort(key=lambda item: (str(item.get("model", {})), str(item.get("context_length", ""))))
    return {
        "schema_version": 1,
        "kind": "a2p_public_profile_metadata",
        "generated_at_utc": utc_timestamp(),
        "metadata_run_count": len(flattened_runs),
        "runs": flattened_runs,
        "raw_artifact_policy": "Chrome trace、Nsight report 和完整 timeline 保留本地；本文件不复制这些原始内容。",
    }


def memory_metadata_payload(data: MemoryData) -> dict[str, JsonValue]:
    """Use cleaned local metadata with an explicit public-artifact policy."""

    if data.metadata is None:
        raise SubmissionPreparationError("memory metadata 不存在")
    payload = dict(data.metadata)
    payload["submission_artifact_policy"] = "仅提交轻量 CSV/JSON；不复制 memory snapshot、pickle 或 Chrome trace。"
    return payload


def build_report_data(
    benchmark: BenchmarkData,
    profile: ProfileData,
    mixed_precision: MixedPrecisionData,
    memory: MemoryData,
) -> dict[str, JsonValue]:
    """Create one compact data source for tables and prose in the Markdown report."""

    return {
        "schema_version": 1,
        "kind": "a2p_report_data",
        "generated_at_utc": utc_timestamp(),
        "privacy": {
            "policy": "仅保留轻量汇总；移除路径、主机、账号、IP、UUID、进程和凭据字段。",
            "raw_artifacts_not_copied": ["Chrome trace", ".nsys-rep", "memory snapshot", "pickle"],
        },
        "benchmark": benchmark_report(benchmark),
        "profile": profile_report(profile),
        "mixed_precision": mixed_precision_report(mixed_precision),
        "memory": memory_report(memory),
    }


def require_complete_inputs(
    benchmark: BenchmarkData,
    profile: ProfileData,
    mixed_precision: MixedPrecisionData,
    memory: MemoryData,
) -> None:
    """Fail closed when the caller requests a complete A2-P public package."""

    missing: list[str] = []
    if not benchmark.rows:
        missing.append("benchmark.csv")
    if not profile.trace_rows:
        missing.append("profile/trace_summary.csv")
    if not profile.metadata_documents:
        missing.append("profile/run_metadata.json")
    if mixed_precision.document is None:
        missing.append("mixed_precision.json")
    if not memory.rows:
        missing.append("memory/peaks.csv")
    if memory.metadata is None:
        missing.append("memory/run_metadata.json")
    if missing:
        raise SubmissionPreparationError("--strict 缺少必交轻量结果：" + ", ".join(missing))


def validate_artifact_path(path: Path) -> None:
    """Ensure generated artifacts cannot escape the selected output root."""

    if path.is_absolute() or not path.parts or any(part == ".." for part in path.parts):
        raise SubmissionPreparationError("生成文件路径必须位于 output root 内")


def build_artifacts(
    benchmark: BenchmarkData,
    profile: ProfileData,
    mixed_precision: MixedPrecisionData,
    memory: MemoryData,
    report_data_path: Path | None,
) -> list[Artifact]:
    """Materialize only intended CSV/JSON content in memory before any write."""

    artifacts: list[Artifact] = []
    if benchmark.rows:
        artifacts.append(Artifact(Path("benchmark.csv"), csv_bytes(benchmark.fields, benchmark.rows)))
    if profile.trace_rows:
        artifacts.append(Artifact(Path("profile") / "trace_summary.csv", csv_bytes(TRACE_COLUMNS, profile.trace_rows)))
    if profile.metadata_documents:
        artifacts.append(Artifact(Path("profile") / "run_metadata.json", json_bytes(profile_metadata_payload(profile))))
    if mixed_precision.document is not None:
        artifacts.append(Artifact(Path("mixed_precision.json"), json_bytes(mixed_precision.document)))
    if memory.rows:
        artifacts.append(Artifact(Path("memory") / "peaks.csv", csv_bytes(memory.fields, memory.rows)))
    if memory.metadata is not None:
        artifacts.append(Artifact(Path("memory") / "run_metadata.json", json_bytes(memory_metadata_payload(memory))))
    if report_data_path is not None:
        artifacts.append(
            Artifact(
                report_data_path,
                json_bytes(build_report_data(benchmark, profile, mixed_precision, memory)),
            )
        )

    seen_paths: set[Path] = set()
    for artifact in artifacts:
        validate_artifact_path(artifact.relative_path)
        if artifact.relative_path in seen_paths:
            raise SubmissionPreparationError("生成文件路径冲突")
        seen_paths.add(artifact.relative_path)
    return artifacts


def validate_artifact_sizes(artifacts: Sequence[Artifact], max_file_bytes: int, max_total_bytes: int) -> None:
    """Enforce the assignment's public-result size envelope before writing."""

    total_bytes = 0
    for artifact in artifacts:
        size = len(artifact.content)
        if size > max_file_bytes:
            raise SubmissionPreparationError(f"生成文件超过单文件限制：{artifact.relative_path.as_posix()}")
        total_bytes += size
    if total_bytes > max_total_bytes:
        raise SubmissionPreparationError("生成 results 汇总超过总大小限制；可降低 --max-profile-rows-per-run-stage")


def is_within(path: Path, parent: Path) -> bool:
    """Return whether ``path`` is equal to or below ``parent`` after resolution."""

    try:
        path.resolve().relative_to(parent.resolve())
    except ValueError:
        return False
    return True


def validate_output_separation(output_root: Path, sources: SourcePaths) -> None:
    """Prevent accidental in-place replacement of local raw results."""

    if output_root.exists() and output_root.is_symlink():
        raise SubmissionPreparationError("output root 不能是符号链接")
    source_directories = (sources.benchmark.directory, sources.profile_dir, sources.memory_dir)
    for source in source_directories:
        if is_within(output_root, source) or is_within(source, output_root):
            raise SubmissionPreparationError("output root 不能位于任何原始 CSV 输入目录内，也不能包含该目录")
    for source in sources.mixed_inputs:
        if is_within(source, output_root):
            raise SubmissionPreparationError("output root 不能包含 mixed precision 原始输入")


def write_stage(stage: Path, artifacts: Sequence[Artifact]) -> None:
    """Write validated files into a fresh staging directory."""

    for artifact in artifacts:
        destination = stage / artifact.relative_path
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(artifact.content)


def backup_path_for(output_root: Path) -> Path:
    """Choose a same-parent backup name for an explicit overwrite transaction."""

    index = 0
    while True:
        suffix = f".{os.getpid()}.{index}"
        candidate = output_root.with_name(f".{output_root.name}.backup{suffix}")
        if not candidate.exists():
            return candidate
        index += 1


def commit_stage(stage: Path, output_root: Path, overwrite: bool) -> None:
    """Atomically publish a staged directory, with rollback for --overwrite."""

    output_root.parent.mkdir(parents=True, exist_ok=True)
    if not output_root.exists():
        os.replace(stage, output_root)
        return
    if not overwrite:
        raise SubmissionPreparationError("output root 已存在；请先检查内容，确认后才传入 --overwrite")
    if output_root.is_symlink():
        raise SubmissionPreparationError("拒绝覆写符号链接 output root")

    backup = backup_path_for(output_root)
    os.replace(output_root, backup)
    try:
        os.replace(stage, output_root)
    except OSError:
        os.replace(backup, output_root)
        raise
    try:
        shutil.rmtree(backup)
    except OSError as error:
        raise SubmissionPreparationError("已写入新 output root，但无法清理旧备份目录") from error


def make_source_paths(args: argparse.Namespace) -> SourcePaths:
    """Resolve optional CLI overrides relative to the supplied source root."""

    source_root = cast(Path, args.source_root)
    benchmark_dir = cast(Path, args.benchmark_dir) if args.benchmark_dir is not None else source_root / "benchmark"
    explicit_benchmarks = tuple(cast(list[Path], args.benchmark_input)) if args.benchmark_input else None
    mixed_inputs = tuple(cast(list[Path], args.mixed_input)) if args.mixed_input else (source_root / "mixed_precision.json",)
    return SourcePaths(
        benchmark=BenchmarkInputs(directory=benchmark_dir, files=explicit_benchmarks),
        profile_dir=cast(Path, args.profile_dir) if args.profile_dir is not None else source_root / "profile",
        memory_dir=cast(Path, args.memory_dir) if args.memory_dir is not None else source_root / "memory",
        mixed_inputs=mixed_inputs,
    )


def plan_payload(artifacts: Sequence[Artifact]) -> dict[str, JsonValue]:
    """Create a non-identifying dry-run display without local filesystem paths."""

    return {
        "artifact_count": len(artifacts),
        "total_bytes": sum(len(artifact.content) for artifact in artifacts),
        "artifacts": [
            {"path": artifact.relative_path.as_posix(), "bytes": len(artifact.content)} for artifact in artifacts
        ],
    }


def run(args: argparse.Namespace) -> int:
    """Prepare, validate, and optionally publish a public result package."""

    sources = make_source_paths(args)
    output_root = cast(Path, args.output_root)
    validate_output_separation(output_root, sources)

    benchmark = collect_benchmark(sources.benchmark, args.max_input_bytes)
    profile = collect_profile(sources.profile_dir, args.max_input_bytes, args.max_profile_rows_per_run_stage)
    memory = collect_memory(sources.memory_dir, args.max_input_bytes)
    mixed_precision = collect_mixed_precision(sources.mixed_inputs, args.max_input_bytes)
    if args.strict:
        require_complete_inputs(benchmark, profile, mixed_precision, memory)

    artifacts = build_artifacts(benchmark, profile, mixed_precision, memory, args.report_data)
    validate_artifact_sizes(artifacts, args.max_file_bytes, args.max_total_bytes)
    if args.dry_run:
        print(json.dumps(plan_payload(artifacts), ensure_ascii=False, indent=2, sort_keys=True))
        return 0

    if output_root.exists() and not args.overwrite:
        raise SubmissionPreparationError("output root 已存在；默认不覆写。请使用 --dry-run 审核，随后显式传入 --overwrite")

    output_root.parent.mkdir(parents=True, exist_ok=True)
    stage = Path(tempfile.mkdtemp(prefix=f".{output_root.name}.staging-", dir=output_root.parent))
    try:
        write_stage(stage, artifacts)
        commit_stage(stage, output_root, args.overwrite)
    finally:
        if stage.exists():
            shutil.rmtree(stage)

    print(f"已生成 {len(artifacts)} 个公开轻量文件；默认策略未复制 trace、snapshot 或 pickle。")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point with redacted, actionable errors."""

    args = parse_args(argv)
    try:
        return run(args)
    except (OSError, csv.Error, json.JSONDecodeError, SubmissionPreparationError, ValueError) as error:
        print(f"提交结果整理失败：{redact_text(error)}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
