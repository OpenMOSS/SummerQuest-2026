"""Safely repair the incomplete first-round profiling result artifacts.

The offline path rebuilds Task 2 submission artifacts from the retained Chrome
traces.  The H200 path validates those already-published Task 2 artifacts, then
performs only the missing Task 3 numeric diagnostic and Task 4 OOM replays; it
deliberately does not re-run completed benchmark or profile experiments.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import subprocess
import sys
import tempfile
from collections.abc import Iterable
from pathlib import Path
from typing import Any

from profiling.collect_utils import publish_files_transactionally


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
PROFILE_RESULTS = RESULTS / "profile"
PROFILE_TRACES = ROOT / "local_artifacts" / "profile"
MEMORY_RESULTS = RESULTS / "memory"
MEMORY_SNAPSHOTS = ROOT / "local_artifacts" / "memory"
STAGING_ROOT = ROOT / "local_artifacts" / "repair_staging"

PROFILE_RUNS = PROFILE_RESULTS / "runs.jsonl"
PROFILE_SUMMARY = PROFILE_RESULTS / "trace_summary.csv"
PROFILE_METADATA = PROFILE_RESULTS / "run_metadata.json"
MIXED_PRECISION = RESULTS / "mixed_precision.json"
MIXED_PRECISION_BENCHMARKS = RESULTS / "mixed_precision_benchmark.jsonl"
MEMORY_FILES = ("runs.jsonl", "failures.jsonl", "peaks.csv", "run_metadata.json")
OOM_TARGETS = {
    ("xl", 2048, 4, "train_step", "fp32"),
    ("xl", 2048, 4, "train_step", "bf16"),
}
MEMORY_SUCCESS_BASELINE = {
    ("xl", 128, 4, "forward", "fp32"),
    ("xl", 128, 4, "train_step", "fp32"),
    ("xl", 2048, 4, "forward", "fp32"),
    ("xl", 2048, 1, "train_step", "fp32"),
    ("xl", 128, 4, "forward", "bf16"),
    ("xl", 128, 4, "train_step", "bf16"),
    ("xl", 2048, 4, "forward", "bf16"),
    ("xl", 2048, 1, "train_step", "bf16"),
}
NUMERIC_METRICS = (
    "fp32_loss",
    "bf16_loss",
    "loss_abs_diff",
    "loss_relative_diff",
    "logits_max_abs_diff",
    "logits_rmse",
    "logits_relative_l2_error",
    "top1_agreement",
)
NUMERIC_FINITE_FLAGS = (
    "fp32_loss_finite",
    "bf16_loss_finite",
    "fp32_logits_finite",
    "bf16_logits_finite",
    "all_finite",
)
OOM_MEMORY_STAT_FIELDS = (
    "active_bytes",
    "peak_active_bytes",
    "allocated_bytes",
    "peak_allocated_bytes",
    "reserved_bytes",
    "peak_reserved_bytes",
)
OOM_MEMORY_VALUE_FIELDS = (*OOM_MEMORY_STAT_FIELDS, "free_bytes", "total_bytes", "requested_allocation_bytes")
ABSOLUTE_PATH = re.compile(r"(^|[\s\"'=])/(?:[^\s\"']+)")
WINDOWS_ABSOLUTE_PATH = re.compile(r"[A-Za-z]:[\\/]")
PROFILE_PHASES = ("forward", "backward", "optimizer")
ATTENTION_RANGES = ("attention/scores", "attention/softmax", "attention/value")
PROFILE_ENVIRONMENT_FIELDS = ("device_name", "torch_version", "cuda_version", "python_version")


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise ValueError(f"{path}:{line_number} is not valid JSON: {error}") from error
        if not isinstance(record, dict):
            raise ValueError(f"{path}:{line_number} must be a JSON object.")
        records.append(record)
    return records


def run_command(command: list[str], *, dry_run: bool) -> None:
    print("Running:" if not dry_run else "Planned:", " ".join(command), flush=True)
    if not dry_run:
        subprocess.run(command, cwd=ROOT, check=True)


def atomic_publish(pairs: Iterable[tuple[Path, Path]]) -> None:
    """Compatibility wrapper for the shared transactional publisher."""

    publish_files_transactionally(pairs)


def expected_profile_run_names() -> set[str]:
    return {
        f"{model_size}_ctx{context_length}_train_step_fp32"
        for model_size in ("small", "medium")
        for context_length in (256, 512, 1024)
    }


def _contains_absolute_path(value: object) -> bool:
    """Reject Unix/Windows absolute paths while allowing relative artifact names."""

    serialized = json.dumps(value, ensure_ascii=False, sort_keys=True)
    return bool(ABSOLUTE_PATH.search(serialized) or WINDOWS_ABSOLUTE_PATH.search(serialized))


def _finite_number(value: object, *, description: str, allow_none: bool = False) -> float | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        raise ValueError(f"{description} must be a finite number{', or null' if allow_none else ''}.")
    return float(value)


def _nonnegative_int(value: object, *, description: str, allow_none: bool = False) -> int | None:
    if value is None and allow_none:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{description} must be a non-negative integer{', or null' if allow_none else ''}.")
    return value


def _csv_number(row: dict[str, str], field: str, *, description: str) -> float:
    value = row.get(field, "")
    if not value:
        raise ValueError(f"{description} is missing {field}.")
    try:
        parsed = float(value)
    except ValueError as error:
        raise ValueError(f"{description} has a non-numeric {field}: {value!r}.") from error
    if not math.isfinite(parsed) or parsed < 0:
        raise ValueError(f"{description} has an invalid {field}: {value!r}.")
    return parsed


def _csv_positive_int(row: dict[str, str], field: str, *, description: str) -> int:
    value = row.get(field, "")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{description} has an invalid integer {field}: {value!r}.") from error
    if parsed < 1:
        raise ValueError(f"{description} must have at least one {field}.")
    return parsed


def validate_profile_outputs(summary: Path, metadata: Path) -> None:
    with summary.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        rows = list(reader)
    required_columns = {
        "run_name",
        "row_type",
        "stage",
        "name",
        "calls",
        "cpu_time_total_us",
        "cuda_time_total_us",
        "kernel_calls",
        "range_duration_us",
    }
    if not rows or reader.fieldnames is None or not required_columns.issubset(reader.fieldnames):
        raise ValueError("Profile summary is empty or does not have the required measurement-only schema.")
    if {row.get("row_type") for row in rows} != {"range", "cpu_op", "cuda_activity"}:
        raise ValueError("Profile summary does not contain the expected range, CPU-op, and CUDA-activity rows.")
    if any(row.get("stage") == "profile/warmup" or row.get("name") == "profile/warmup" for row in rows):
        raise ValueError("Profile summary unexpectedly includes profile/warmup.")
    if any(row.get("activity_type") == "gpu_user_annotation" for row in rows):
        raise ValueError("Profile summary must not derive CUDA time from gpu_user_annotation rows.")
    summary_runs = {row["run_name"] for row in rows}
    if summary_runs != expected_profile_run_names():
        raise ValueError(f"Expected six profile runs, got {sorted(summary_runs)}.")

    for run_name in sorted(summary_runs):
        run_rows = [row for row in rows if row["run_name"] == run_name]
        measure_rows = [
            row
            for row in run_rows
            if row.get("row_type") == "range" and row.get("name") == "profile/measure" and row.get("stage") == "profile/measure"
        ]
        if len(measure_rows) != 1:
            raise ValueError(f"{run_name}: expected exactly one measurement CPU range.")
        measure = measure_rows[0]
        if _csv_positive_int(measure, "calls", description=f"{run_name} measurement range") != 1:
            raise ValueError(f"{run_name}: profile/measure must represent exactly one trace range.")
        _csv_number(measure, "range_duration_us", description=f"{run_name} measurement range")
        if measure.get("cuda_time_total_us") or measure.get("cpu_time_total_us"):
            raise ValueError(f"{run_name}: a CPU range cannot claim physical CUDA or CPU-op totals.")

        for phase in PROFILE_PHASES:
            phase_rows = [
                row
                for row in run_rows
                if row.get("row_type") == "range" and row.get("name") == phase and row.get("stage") == phase
            ]
            if len(phase_rows) != 1:
                raise ValueError(f"{run_name}: expected exactly one inclusive {phase} range.")
            _csv_positive_int(phase_rows[0], "calls", description=f"{run_name} {phase} range")
            _csv_number(phase_rows[0], "range_duration_us", description=f"{run_name} {phase} range")
            if phase_rows[0].get("inclusive") != "true":
                raise ValueError(f"{run_name}: {phase} range must be marked inclusive.")

        for attention_name in ATTENTION_RANGES:
            attention_rows = [
                row
                for row in run_rows
                if row.get("row_type") == "range" and row.get("name") == attention_name and row.get("stage") == "forward"
            ]
            if len(attention_rows) != 1:
                raise ValueError(f"{run_name}: expected one nested {attention_name} range.")
            _csv_positive_int(attention_rows[0], "calls", description=f"{run_name} {attention_name} range")
            _csv_number(attention_rows[0], "range_duration_us", description=f"{run_name} {attention_name} range")
            if attention_rows[0].get("inclusive") != "true":
                raise ValueError(f"{run_name}: nested {attention_name} must be marked inclusive.")

        cpu_rows = [row for row in run_rows if row.get("row_type") == "cpu_op"]
        cuda_rows = [row for row in run_rows if row.get("row_type") == "cuda_activity"]
        if not cpu_rows or not cuda_rows:
            raise ValueError(f"{run_name}: profile summary must retain both CPU operations and physical CUDA activity.")
        for row in cpu_rows:
            if row.get("stage") not in PROFILE_PHASES or not row.get("name"):
                raise ValueError(f"{run_name}: CPU-op row has no valid phase/name association.")
            _csv_positive_int(row, "calls", description=f"{run_name} CPU-op row")
            _csv_number(row, "cpu_time_total_us", description=f"{run_name} CPU-op row")
            if row.get("cuda_time_total_us") or row.get("range_duration_us"):
                raise ValueError(f"{run_name}: CPU-op row mixes incompatible time domains.")
        for row in cuda_rows:
            if row.get("stage") not in PROFILE_PHASES or row.get("activity_type") not in {"kernel", "gpu_memcpy", "gpu_memset"}:
                raise ValueError(f"{run_name}: CUDA row has no valid physical activity attribution.")
            calls = _csv_positive_int(row, "calls", description=f"{run_name} CUDA row")
            _csv_number(row, "cuda_time_total_us", description=f"{run_name} CUDA row")
            if row.get("cpu_time_total_us") or row.get("range_duration_us"):
                raise ValueError(f"{run_name}: CUDA row mixes incompatible time domains.")
            kernel_calls = _csv_number(row, "kernel_calls", description=f"{run_name} CUDA row")
            expected_kernel_calls = calls if row["activity_type"] == "kernel" else 0
            if kernel_calls != expected_kernel_calls:
                raise ValueError(f"{run_name}: CUDA kernel Calls do not match the activity type.")

    loaded_metadata = json.loads(metadata.read_text(encoding="utf-8"))
    if not isinstance(loaded_metadata, list) or len(loaded_metadata) != 6:
        raise ValueError("Profile metadata must contain exactly six runs.")
    metadata_runs = {entry.get("run_name") for entry in loaded_metadata if isinstance(entry, dict)}
    if metadata_runs != expected_profile_run_names():
        raise ValueError(f"Profile metadata run names do not match the expected matrix: {metadata_runs}.")
    for entry in loaded_metadata:
        if not isinstance(entry, dict):
            raise ValueError("Every profile metadata entry must be an object.")
        run_name = entry.get("run_name")
        if not isinstance(run_name, str):
            raise ValueError("Profile metadata has a non-string run name.")
        if entry.get("model_size") not in {"small", "medium"} or entry.get("context_length") not in {256, 512, 1024}:
            raise ValueError(f"{run_name}: invalid model/context metadata.")
        if entry.get("batch_size") != 4 or entry.get("mode") != "train_step" or entry.get("dtype") != "fp32":
            raise ValueError(f"{run_name}: metadata does not describe the required Task 2 run configuration.")
        if entry.get("warmup_steps") != 5 or entry.get("measurement_steps") != 1 or entry.get("tool") != "torch.profiler":
            raise ValueError(f"{run_name}: metadata has an invalid measurement protocol.")
        protocol = entry.get("warmup_protocol")
        if protocol != {"outside_profiler_steps": 4, "inside_profiler_steps": 1}:
            raise ValueError(f"{run_name}: metadata has an invalid warm-up protocol.")
        if entry.get("trace_file") != f"{run_name}.json" or entry.get("summary_file") != "trace_summary.csv":
            raise ValueError(f"{run_name}: metadata references an unexpected public artifact.")
        if not isinstance(entry.get("command"), str) or not entry["command"]:
            raise ValueError(f"{run_name}: metadata is missing a sanitized command.")
        environment = entry.get("environment")
        if not isinstance(environment, dict) or any(not isinstance(environment.get(field), str) or not environment[field] for field in PROFILE_ENVIRONMENT_FIELDS):
            raise ValueError(f"{run_name}: metadata environment is incomplete.")
        if _contains_absolute_path(entry):
            raise ValueError("Profile metadata contains an absolute local path.")


def rebuild_profile(*, stage: Path, dry_run: bool) -> tuple[tuple[Path, Path], ...]:
    if not dry_run:
        stage.mkdir(parents=True, exist_ok=True)
    staged_summary = stage / "trace_summary.csv"
    staged_metadata = stage / "run_metadata.json"
    command = [
        sys.executable,
        "profiling/trace_summary.py",
        "--trace-dir",
        str(PROFILE_TRACES),
        "--runs",
        str(PROFILE_RUNS),
        "--output",
        str(staged_summary),
        "--metadata-output",
        str(staged_metadata),
    ]
    run_command(command, dry_run=dry_run)
    if dry_run:
        return ()
    validate_profile_outputs(staged_summary, staged_metadata)
    return ((staged_summary, PROFILE_SUMMARY), (staged_metadata, PROFILE_METADATA))


def preflight_profile_inputs() -> None:
    """Validate the retained trace/audit inputs without creating an output file."""

    command = [
        sys.executable,
        "profiling/trace_summary.py",
        "--trace-dir",
        str(PROFILE_TRACES),
        "--runs",
        str(PROFILE_RUNS),
        "--check",
    ]
    print("Preflight:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def validate_existing_task2_for_h200() -> None:
    """Require a valid local Task 2 publication without requiring raw traces.

    Raw Chrome traces are intentionally ignored by Git, whereas the repaired
    CSV and metadata are public result artifacts.  The H200-only repair must
    not force users to copy retained traces merely to fill the independent
    Task 3 and Task 4 gaps.
    """

    try:
        validate_profile_outputs(PROFILE_SUMMARY, PROFILE_METADATA)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise ValueError(
            "--run-h200-repairs only repairs Task 3 and Task 4, but the existing Task 2 "
            f"publication is not valid: {error}. Run --offline on a machine that retains the six Chrome traces."
        ) from error
    print("Task 2 publication preflight passed; retained Chrome traces will not be read or re-run.", flush=True)


def require_h200(*, allow_other_cuda: bool) -> None:
    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("--run-h200-repairs requires a CUDA-capable PyTorch environment.")
    device = torch.device("cuda", torch.cuda.current_device())
    if not torch.cuda.is_bf16_supported(device):
        raise RuntimeError("--run-h200-repairs requires CUDA BF16 autocast support.")
    device_name = torch.cuda.get_device_name(device)
    if "H200" not in device_name and not allow_other_cuda:
        raise RuntimeError(f"Expected an H200 for result continuity, found {device_name!r}. Use --allow-other-cuda to override explicitly.")
    print(f"CUDA preflight passed: {device_name}", flush=True)


def validate_numeric_trend(path: Path) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("mixed_precision.json must be a JSON object.")
    if not isinstance(payload.get("accumulation"), dict) or not isinstance(payload.get("toy_bf16"), dict):
        raise ValueError("mixed_precision.json no longer preserves the accumulation and ToyModel observations.")
    trend = payload.get("language_model_numeric_trend") if isinstance(payload, dict) else None
    if not isinstance(trend, dict):
        raise ValueError("mixed_precision.json is missing language_model_numeric_trend.")
    configuration = trend.get("configuration")
    if not isinstance(configuration, dict):
        raise ValueError("Numeric trend is missing its fixed configuration.")
    required_configuration = {
        "model_size": "small",
        "batch_size": 4,
        "context_length": 512,
        "seed": 0,
        "warmup_steps": 5,
        "measurement_steps": 10,
        "mode": "train_step",
        "fp32_precision": "fp32",
        "bf16_precision": "bf16_autocast",
    }
    if any(configuration.get(name) != value for name, value in required_configuration.items()):
        raise ValueError("Numeric trend does not use the required fixed FP32/BF16 protocol.")
    model_configuration = configuration.get("model_config")
    if not isinstance(model_configuration, dict) or model_configuration.get("batch_size") != 4 or model_configuration.get("context_length") != 512:
        raise ValueError("Numeric trend model configuration does not match batch 4/context 512.")
    comparison = trend.get("comparison")
    if not isinstance(comparison, dict) or comparison.get("initialization") != "shared_cpu_fp32_state_dict":
        raise ValueError("Numeric trend does not document shared CPU FP32 initialization.")
    environment = trend.get("environment")
    if not isinstance(environment, dict) or any(not isinstance(environment.get(field), str) or not environment[field] for field in PROFILE_ENVIRONMENT_FIELDS):
        raise ValueError("Numeric trend environment is incomplete.")
    steps = trend.get("steps")
    if not isinstance(steps, list) or len(steps) != 10:
        raise ValueError("Numeric trend must contain exactly ten measurement steps.")
    for index, step in enumerate(steps, start=1):
        if not isinstance(step, dict):
            raise ValueError(f"Numeric trend step {index} must be an object.")
        if step.get("measurement_step") != index or step.get("global_step") != index + 4:
            raise ValueError(f"Numeric trend step {index} has an unexpected measurement/global-step index.")
        for metric in NUMERIC_METRICS:
            value = _finite_number(step.get(metric), description=f"Numeric trend step {index} {metric}", allow_none=True)
            if metric == "top1_agreement" and value is not None and not 0.0 <= value <= 1.0:
                raise ValueError(f"Numeric trend step {index} has an invalid top-1 agreement.")
        for flag in NUMERIC_FINITE_FLAGS:
            if not isinstance(step.get(flag), bool):
                raise ValueError(f"Numeric trend step {index} is missing Boolean {flag}.")

    summary = trend.get("summary")
    if not isinstance(summary, dict):
        raise ValueError("Numeric trend is missing its summary.")
    if summary.get("measurement_steps") != 10 or not isinstance(summary.get("all_steps_finite"), bool):
        raise ValueError("Numeric trend summary has an invalid step count or finite-value flag.")
    if summary["all_steps_finite"] != all(step["all_finite"] for step in steps):
        raise ValueError("Numeric trend summary has an incorrect all-steps-finite flag.")
    finite_counts = summary.get("finite_step_counts")
    if not isinstance(finite_counts, dict):
        raise ValueError("Numeric trend summary is missing finite-value counts.")
    for flag in NUMERIC_FINITE_FLAGS:
        expected_count = sum(bool(step[flag]) for step in steps)
        if finite_counts.get(flag) != expected_count:
            raise ValueError(f"Numeric trend summary has an invalid {flag} count.")
    metric_summary = summary.get("metrics")
    if not isinstance(metric_summary, dict):
        raise ValueError("Numeric trend summary is missing metric extrema.")
    for metric in NUMERIC_METRICS:
        aggregate = metric_summary.get(metric)
        if not isinstance(aggregate, dict):
            raise ValueError(f"Numeric trend summary is missing {metric} extrema.")
        values = [float(step[metric]) for step in steps if isinstance(step.get(metric), int | float) and not isinstance(step[metric], bool)]
        expected = {
            "min": min(values) if values else None,
            "mean": sum(values) / len(values) if values else None,
            "max": max(values) if values else None,
            "available_steps": len(values),
        }
        if aggregate.get("available_steps") != expected["available_steps"]:
            raise ValueError(f"Numeric trend summary has an invalid {metric} available-step count.")
        for name in ("min", "mean", "max"):
            actual = _finite_number(aggregate.get(name), description=f"Numeric trend summary {metric} {name}", allow_none=True)
            expected_value = expected[name]
            if expected_value is None:
                if actual is not None:
                    raise ValueError(f"Numeric trend summary should report null {metric} {name}.")
            elif actual is None or not math.isclose(actual, expected_value, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Numeric trend summary has an incorrect {metric} {name}.")

    benchmark_records = read_jsonl(MIXED_PRECISION_BENCHMARKS)
    if len(benchmark_records) != 20:
        raise ValueError("The existing 20 mixed-precision performance records were not preserved.")
    for record in benchmark_records:
        if not isinstance(record.get("model_config"), dict) or not isinstance(record.get("run_config"), dict) or not isinstance(record.get("statistics"), dict):
            raise ValueError("A preserved mixed-precision performance record has an invalid schema.")
        if _contains_absolute_path(record):
            raise ValueError("A preserved mixed-precision performance record contains an absolute path.")
    if _contains_absolute_path(payload):
        raise ValueError("mixed_precision.json contains an absolute path.")


def collect_numeric_trend(*, stage: Path, dry_run: bool) -> tuple[tuple[Path, Path], ...]:
    if not dry_run:
        stage.mkdir(parents=True, exist_ok=True)
    staged_output = stage / "mixed_precision.json"
    if not dry_run:
        shutil.copy2(MIXED_PRECISION, staged_output)
    command = [sys.executable, "profiling/mixed_precision.py", "numeric-trend", "--output", str(staged_output)]
    run_command(command, dry_run=dry_run)
    if dry_run:
        return ()
    validate_numeric_trend(staged_output)
    return ((staged_output, MIXED_PRECISION),)


def target_identity(record: dict[str, Any]) -> tuple[str, int, int, str, str] | None:
    try:
        model_size = record["model_size"]
        context_length = record["context_length"]
        batch_size = record["batch_size"]
        mode = record["mode"]
        dtype = record["dtype"]
    except KeyError:
        return None
    if (
        not isinstance(model_size, str)
        or not isinstance(mode, str)
        or not isinstance(dtype, str)
        or isinstance(context_length, bool)
        or not isinstance(context_length, int)
        or isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
    ):
        return None
    return (model_size, context_length, batch_size, mode, dtype)


def success_identity(record: dict[str, Any]) -> tuple[str, int, int, str, str] | None:
    model = record.get("model_config")
    run = record.get("run_config")
    if not isinstance(model, dict) or not isinstance(run, dict):
        return None
    return target_identity(
        {
            "model_size": run.get("model_size"),
            "context_length": model.get("context_length"),
            "batch_size": model.get("batch_size"),
            "mode": run.get("mode"),
            "dtype": run.get("precision"),
        }
    )


def _validate_oom_failure(failure: dict[str, Any], *, target: tuple[str, int, int, str, str]) -> str:
    """Validate either structured or explicitly-unavailable CUDA OOM telemetry."""

    if failure.get("exception") != "cuda_oom":
        raise ValueError(f"OOM replay failed unexpectedly for {target}: {failure.get('exception')}.")
    availability = failure.get("oom_telemetry_available")
    if not isinstance(availability, bool):
        raise ValueError(f"OOM replacement for {target} does not declare telemetry availability.")
    scope = failure.get("failure_scope")
    phase = failure.get("failure_phase")
    peak_scope = failure.get("peak_scope")
    if scope not in {"initialization", "warmup", "measurement", "unavailable"}:
        raise ValueError(f"OOM replacement for {target} has an invalid failure scope.")
    if phase not in {"forward", "backward", "optimizer", None}:
        raise ValueError(f"OOM replacement for {target} has an invalid failure phase.")
    expected_peaks = {
        "initialization": "initialization",
        "warmup": "warmup",
        "measurement": "post_warmup_measurement",
        "unavailable": "unavailable",
    }
    if peak_scope != expected_peaks[scope]:
        raise ValueError(f"OOM replacement for {target} has an invalid peak scope.")
    memory = failure.get("memory")
    if not isinstance(memory, dict):
        raise ValueError(f"OOM replacement for {target} has no structured memory summary.")
    telemetry_status = memory.get("telemetry_status")
    if telemetry_status not in {"available", "partial", "unavailable"}:
        raise ValueError(f"OOM replacement for {target} has an invalid telemetry status.")
    raw_statistics = memory.get("statistics_bytes")
    if not isinstance(raw_statistics, dict):
        raise ValueError(f"OOM replacement for {target} has no allocator statistics object.")
    values: dict[str, int | None] = {}
    for field in OOM_MEMORY_STAT_FIELDS:
        values[field] = _nonnegative_int(raw_statistics.get(field), description=f"OOM {target} {field}", allow_none=True)
    for field in ("free_bytes", "total_bytes", "requested_allocation_bytes"):
        values[field] = _nonnegative_int(memory.get(field), description=f"OOM {target} {field}", allow_none=True)
    unavailable_fields = memory.get("unavailable_fields")
    if not isinstance(unavailable_fields, list) or set(unavailable_fields) != {field for field, value in values.items() if value is None}:
        raise ValueError(f"OOM replacement for {target} has inconsistent unavailable telemetry fields.")
    available_values = sum(value is not None for value in values.values())
    if availability:
        if telemetry_status not in {"available", "partial"} or available_values == 0:
            raise ValueError(f"OOM replacement for {target} misreports available telemetry.")
    else:
        if telemetry_status != "unavailable" or available_values != 0:
            raise ValueError(f"OOM replacement for {target} does not honestly mark unavailable telemetry.")
    environment = failure.get("environment")
    if not isinstance(environment, dict) or any(field not in environment for field in PROFILE_ENVIRONMENT_FIELDS):
        raise ValueError(f"OOM replacement for {target} has no public environment summary.")
    if any(value is not None and not isinstance(value, str) for value in environment.values()):
        raise ValueError(f"OOM replacement for {target} has an invalid public environment summary.")
    forbidden = {"stderr", "stdout", "raw_error", "error_text", "exception_text", "pid", "traceback"}
    if forbidden.intersection(failure):
        raise ValueError(f"OOM replacement for {target} leaks raw failure details.")
    if _contains_absolute_path(failure):
        raise ValueError(f"OOM replacement for {target} contains an absolute path.")
    return telemetry_status


def validate_memory_repair(memory_directory: Path) -> None:
    failures = read_jsonl(memory_directory / "failures.jsonl")
    successes = read_jsonl(memory_directory / "runs.jsonl")
    success_targets: set[tuple[str, int, int, str, str]] = set()
    for record in successes:
        identity = success_identity(record)
        if identity is None:
            raise ValueError("A memory success record has an invalid configuration schema.")
        if identity in success_targets:
            raise ValueError(f"Memory success records contain duplicate configuration {identity}.")
        success_targets.add(identity)
        if _contains_absolute_path(record):
            raise ValueError("A memory success record contains an absolute path.")
    if not MEMORY_SUCCESS_BASELINE.issubset(success_targets):
        raise ValueError("The existing eight successful Task 4 runs were not preserved.")

    failure_targets: dict[tuple[str, int, int, str, str], dict[str, Any]] = {}
    for record in failures:
        identity = target_identity(record)
        if identity not in OOM_TARGETS:
            continue
        if identity in failure_targets:
            raise ValueError(f"OOM replay contains duplicate failure records for {identity}.")
        failure_targets[identity] = record
    for target in OOM_TARGETS:
        if target in success_targets:
            if target in failure_targets:
                raise ValueError(f"OOM replay retained a stale failure after success for {target}.")
            continue
        failure = failure_targets.get(target)
        if failure is None:
            raise ValueError(f"OOM replay produced neither a success nor failure record for {target}.")
        _validate_oom_failure(failure, target=target)

    peaks_path = memory_directory / "peaks.csv"
    with peaks_path.open(encoding="utf-8", newline="") as input_file:
        reader = csv.DictReader(input_file)
        peak_rows = list(reader)
    peak_columns = {"model_size", "mode", "dtype", "batch_size", "context_length", *OOM_MEMORY_STAT_FIELDS}
    if reader.fieldnames is None or not peak_columns.issubset(reader.fieldnames) or len(peak_rows) != len(successes):
        raise ValueError("Memory peaks.csv does not match the preserved success records.")
    metadata = json.loads((memory_directory / "run_metadata.json").read_text(encoding="utf-8"))
    if not isinstance(metadata, list) or len(metadata) != len(successes):
        raise ValueError("Memory metadata does not match the preserved success records.")
    if _contains_absolute_path(metadata):
        raise ValueError("Memory metadata contains an absolute path.")


def copy_memory_inputs(destination: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    for name in MEMORY_FILES:
        source = MEMORY_RESULTS / name
        if not source.exists():
            raise ValueError(f"Required memory artifact is missing: {source}")
        shutil.copy2(source, destination / name)


def validate_historical_oom_targets(memory_directory: Path) -> None:
    """Ensure the H200 entry point can only replay the two agreed OOM cases."""

    targets: list[tuple[str, int, int, str, str]] = []
    for record in read_jsonl(memory_directory / "failures.jsonl"):
        if record.get("exception") != "cuda_oom":
            continue
        identity = target_identity(record)
        if identity is None:
            raise ValueError("A historical CUDA OOM record has an invalid configuration.")
        targets.append(identity)
    if set(targets) != OOM_TARGETS or len(targets) != len(OOM_TARGETS):
        raise ValueError(
            "The H200 repair may replay only the two agreed XL/context-2048/batch-4/train-step OOMs; "
            f"found {sorted(targets)}."
        )


def staged_success_snapshot_pairs(*, memory_directory: Path, snapshot_directory: Path) -> tuple[tuple[Path, Path], ...]:
    """Publish snapshots only when a replay unexpectedly succeeds and references one."""

    pairs: list[tuple[Path, Path]] = []
    for record in read_jsonl(memory_directory / "runs.jsonl"):
        if success_identity(record) not in OOM_TARGETS:
            continue
        memory = record.get("memory")
        snapshot_file = memory.get("snapshot_file") if isinstance(memory, dict) else None
        if not isinstance(snapshot_file, str) or Path(snapshot_file).name != snapshot_file:
            raise ValueError("An unexpected OOM-retry success has no safe snapshot filename.")
        staged_snapshot = snapshot_directory / snapshot_file
        if not staged_snapshot.is_file():
            raise ValueError("An unexpected OOM-retry success is missing its staged memory snapshot.")
        pairs.append((staged_snapshot, MEMORY_SNAPSHOTS / snapshot_file))
    return tuple(pairs)


def replay_ooms(*, stage: Path, dry_run: bool) -> tuple[tuple[Path, Path], ...]:
    staged_memory = stage
    staged_snapshots = stage / "snapshots"
    input_directory = MEMORY_RESULTS if dry_run else staged_memory
    if not dry_run:
        copy_memory_inputs(staged_memory)
    validate_historical_oom_targets(input_directory)
    target_labels = ", ".join(f"{dtype.upper()} XL/context-{context}/batch-{batch}/{mode}" for _, context, batch, mode, dtype in sorted(OOM_TARGETS))
    print(f"OOM replay boundary: exactly {len(OOM_TARGETS)} historical targets ({target_labels}).", flush=True)
    command = [
        sys.executable,
        "profiling/collect_memory.py",
        "--output-dir",
        str(staged_memory),
        "--snapshot-dir",
        str(staged_snapshots),
        "--retry-oom",
    ]
    run_command(command, dry_run=dry_run)
    if dry_run:
        return ()
    validate_memory_repair(staged_memory)
    memory_pairs = tuple((staged_memory / name, MEMORY_RESULTS / name) for name in MEMORY_FILES)
    return memory_pairs + staged_success_snapshot_pairs(memory_directory=staged_memory, snapshot_directory=staged_snapshots)


def memory_outcome_status(memory_directory: Path) -> str:
    """Summarize the two target outcomes after strict validation has passed."""

    successes = {success_identity(record) for record in read_jsonl(memory_directory / "runs.jsonl")}
    failures = {
        target_identity(record): record
        for record in read_jsonl(memory_directory / "failures.jsonl")
        if target_identity(record) in OOM_TARGETS
    }
    succeeded = sum(target in successes for target in OOM_TARGETS)
    structured = sum(
        target not in successes and failures[target].get("oom_telemetry_available") is True
        for target in OOM_TARGETS
    )
    unavailable = len(OOM_TARGETS) - succeeded - structured
    parts: list[str] = []
    if succeeded:
        parts.append(f"{succeeded} unexpected success{'es' if succeeded != 1 else ''} merged")
    if structured:
        parts.append(f"{structured} structured telemetry")
    if unavailable:
        parts.append(f"{unavailable} telemetry unavailable (honestly marked)")
    return f"complete ({len(OOM_TARGETS)}/{len(OOM_TARGETS)} replay outcomes; " + ", ".join(parts) + ")"


def repair_status() -> dict[str, str]:
    status: dict[str, str] = {}
    try:
        validate_profile_outputs(PROFILE_SUMMARY, PROFILE_METADATA)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        status["Task 2 profile traces"] = f"pending (0/6 validated: {error})"
        status["Task 2 metadata environments"] = f"pending (0/6 complete: {error})"
    else:
        status["Task 2 profile traces"] = "complete (6/6 measurement-only traces)"
        status["Task 2 metadata environments"] = "complete (6/6 H200/CUDA/PyTorch/Python fields)"

    try:
        validate_numeric_trend(MIXED_PRECISION)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        status["Task 3 numeric trend"] = f"pending (0/10 validated: {error})"
    else:
        status["Task 3 numeric trend"] = "complete (10/10 paired FP32-vs-BF16 steps; 20 performance records preserved)"

    try:
        validate_memory_repair(MEMORY_RESULTS)
    except (OSError, ValueError, json.JSONDecodeError) as error:
        status["Task 4 OOM telemetry"] = f"pending (0/2 validated: {error})"
    else:
        status["Task 4 OOM telemetry"] = memory_outcome_status(MEMORY_RESULTS)
    status["Screenshots and writeup"] = "pending (intentionally outside this repair script)"
    return status


def print_status() -> None:
    print("\nRepair status:", flush=True)
    for name, value in repair_status().items():
        print(f"  - {name}: {value}", flush=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Repair first-round profiling results without re-running completed experiments.")
    actions = parser.add_mutually_exclusive_group(required=True)
    actions.add_argument("--dry-run", action="store_true", help="Validate existing Task 2 results and print the Task 3/4 H200 repair sequence without writing files or requiring CUDA.")
    actions.add_argument("--offline", action="store_true", help="Rebuild only profile summary and metadata from local traces.")
    actions.add_argument("--run-h200-repairs", action="store_true", help="On H200, repair only Task 3 numeric trend and the two Task 4 OOM records; retain validated Task 2 artifacts.")
    actions.add_argument("--status", action="store_true", help="Read existing result artifacts and print repair completion status.")
    parser.add_argument("--allow-other-cuda", action="store_true", help="Permit --run-h200-repairs on a non-H200 CUDA GPU; the actual environment remains recorded.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.status:
        print_status()
        return 0

    if args.dry_run:
        print("Dry run: no files will be created or changed; only Task 3/4 H200 work is shown.", flush=True)
        validate_existing_task2_for_h200()
        # This literal placeholder avoids even a temporary directory write in dry-run mode.
        stage = STAGING_ROOT / "DRY_RUN"
        collect_numeric_trend(stage=stage / "mixed", dry_run=True)
        replay_ooms(stage=stage / "memory", dry_run=True)
        return 0

    if args.offline:
        preflight_profile_inputs()
        STAGING_ROOT.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(prefix="repair-", dir=STAGING_ROOT) as temporary_directory:
            stage = Path(temporary_directory)
            profile_pairs = rebuild_profile(stage=stage / "profile", dry_run=False)
            atomic_publish(profile_pairs)
        print("Published measurement-only profile summary and environment-complete metadata.", flush=True)
        print_status()
        return 0

    validate_existing_task2_for_h200()
    STAGING_ROOT.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="repair-", dir=STAGING_ROOT) as temporary_directory:
        stage = Path(temporary_directory)
        require_h200(allow_other_cuda=args.allow_other_cuda)
        numeric_pairs = collect_numeric_trend(stage=stage / "mixed", dry_run=False)
        memory_pairs = replay_ooms(stage=stage / "memory", dry_run=False)
        atomic_publish((*numeric_pairs, *memory_pairs))
        print("Published the validated Task 3 numeric-trend and Task 4 selective OOM-repair artifacts.", flush=True)
    print_status()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError, subprocess.CalledProcessError) as error:
        print(f"Repair failed: {error}", file=sys.stderr)
        print_status()
        raise SystemExit(1) from error
