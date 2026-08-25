#!/usr/bin/env python3
"""Validate private A2-P evidence and build the small public result bundle.

The input directory contains authoritative JSON emitted by ``benchmark.py``,
``mixed_precision.py`` and ``memory_snapshot.py`` together with local profiler
traces and allocator snapshots.  This program deliberately publishes only an
allow-listed projection of those inputs.  Full Chrome traces, snapshots and
source timelines never cross the raw/public boundary.

The output is deterministic for identical inputs: rows have fixed ordering,
JSON keys are sorted, timestamps are omitted, and figures use fixed sizes and
metadata.  ``--dry-run`` performs the same validation/rendering/size/privacy
checks in a temporary directory without touching the requested public root.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import csv
from dataclasses import dataclass
import io
import json
import math
import os
from pathlib import Path
import re
import shutil
import statistics
import tempfile
from typing import Any


STARTER_COMMIT = "ca8bc81a59b70516f7ebb2da4808daade877c736"
HANDOUT_VERSION = "26.1.4-rc.3"
ATTACHMENT_BUDGET_BYTES = 2 * 1024 * 1024
MAX_EVIDENCE_JSON_BYTES = 256 * 1024 * 1024
MAX_PROFILE_KERNEL_ROWS_PER_RUN = 40
MAX_PROFILE_UNIQUE_KERNEL_NAMES = 20_000
MAX_PROFILE_KERNEL_NAME_LENGTH = 4_096

BENCHMARK_SCHEMA = "cs336.a2p.benchmark.v1"
MIXED_SCHEMA = "cs336.a2p.mixed-precision.v1"
MEMORY_SCHEMA = "cs336.a2p.memory-snapshot.v1"
RUN_SUITE_SCHEMA = "cs336.a2p.run-suite.v1"
PUBLIC_SCHEMA = "cs336.a2p.public-summary.v1"

MODEL_DIMENSIONS: dict[str, dict[str, int]] = {
    "small": {"d_model": 768, "d_ff": 3_072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1_024, "d_ff": 4_096, "num_layers": 24, "num_heads": 16},
    "large": {"d_model": 1_280, "d_ff": 5_120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2_560, "d_ff": 10_240, "num_layers": 32, "num_heads": 32},
    "10b": {"d_model": 4_608, "d_ff": 12_288, "num_layers": 50, "num_heads": 36},
}
PROFILE_MODELS = ("small", "xl")
PROFILE_CONTEXTS = (256, 512, 1_024)
MIXED_MODELS = ("small", "medium", "large", "xl", "10b")
MEMORY_DTYPES = ("fp32", "bf16")
MEMORY_MODES = ("forward", "train_step")
MEMORY_CONTEXTS = (128, 2_048)
REQUIRED_RANGES = (
    "profile/warmup",
    "profile/measure",
    "forward",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
)

PUBLIC_FILES = (
    "results/benchmark.csv",
    "results/profile/trace_summary.csv",
    "results/profile/run_metadata.json",
    "results/mixed_precision.json",
    "results/memory/peaks.csv",
    "results/memory/run_metadata.json",
    "assets/compute_profile.png",
    "assets/memory_forward_active_timeline.png",
    "assets/memory_train_step_active_timeline.png",
)
PUBLIC_ASSETS = frozenset(path for path in PUBLIC_FILES if path.startswith("assets/"))

BENCHMARK_FIELDS = (
    "run_id",
    "model_size",
    "batch_size",
    "context_length",
    "dtype",
    "mode",
    "warmup_steps",
    "measurement_steps",
    "measurement_step",
    "total_ms",
    "zero_grad_ms",
    "forward_ms",
    "loss_ms",
    "backward_ms",
    "optimizer_ms",
    "loss",
    "mean_ms",
    "sample_std_ms",
    "cv",
    "cv_percent",
    "peak_allocated_bytes",
    "peak_active_bytes",
    "peak_reserved_bytes",
)

PROFILE_FIELDS = (
    "run_id",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "tool",
    "command",
    "scope",
    "op_or_kernel",
    "evidence_source",
    "name",
    "calls",
    "cpu_time_us",
    "cuda_time_us",
    "attribution",
)

MEMORY_FIELDS = (
    "case_id",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "status",
    "is_fallback",
    "fallback_for",
    "warmup_completed",
    "measurement_steps_completed",
    "measured_elapsed_ms",
    "active_bytes_current",
    "active_bytes_peak",
    "allocated_bytes_current",
    "allocated_bytes_peak",
    "reserved_bytes_current",
    "reserved_bytes_peak",
    "max_memory_allocated",
    "maximum_single_allocation_bytes",
    "residual_stream_elements",
    "residual_stream_theory_formula",
    "residual_stream_fp32_bytes",
    "residual_stream_bf16_bytes",
    "exception_type",
    "failure_stage",
)


class SummaryError(RuntimeError):
    """Raised when raw evidence cannot support a complete public submission."""


@dataclass(frozen=True)
class Evidence:
    path: Path
    payload: dict[str, Any]


@dataclass(frozen=True)
class BuildSummary:
    benchmark_samples: int
    profile_runs: int
    profile_rows: int
    mixed_models: int
    memory_cases: int
    attachment_bytes: int
    files: tuple[str, ...] = PUBLIC_FILES

    def as_dict(self) -> dict[str, Any]:
        return {
            "status": "ok",
            "benchmark_samples": self.benchmark_samples,
            "profile_runs": self.profile_runs,
            "profile_rows": self.profile_rows,
            "mixed_models": self.mixed_models,
            "memory_cases": self.memory_cases,
            "attachment_bytes": self.attachment_bytes,
            "attachment_budget_bytes": ATTACHMENT_BUDGET_BYTES,
            "files": list(self.files),
        }


def _expect(condition: bool, message: str) -> None:
    if not condition:
        raise SummaryError(message)


def _mapping(value: Any, label: str) -> dict[str, Any]:
    _expect(isinstance(value, dict), f"{label} must be an object")
    return value


def _sequence(value: Any, label: str) -> list[Any]:
    _expect(isinstance(value, list), f"{label} must be an array")
    return value


def _string(value: Any, label: str, *, allow_empty: bool = False) -> str:
    _expect(isinstance(value, str), f"{label} must be a string")
    _expect(allow_empty or bool(value), f"{label} must not be empty")
    return value


def _integer(value: Any, label: str, *, minimum: int | None = None) -> int:
    _expect(type(value) is int, f"{label} must be an integer")
    if minimum is not None:
        _expect(value >= minimum, f"{label} must be >= {minimum}")
    return value


def _number(value: Any, label: str, *, positive: bool = False, optional: bool = False) -> float | None:
    if value in (None, "") and optional:
        return None
    _expect(type(value) in (int, float, str), f"{label} must be numeric")
    try:
        result = float(value)
    except ValueError as error:
        raise SummaryError(f"{label} must be numeric") from error
    _expect(math.isfinite(result), f"{label} must be finite")
    if positive:
        _expect(result > 0.0, f"{label} must be positive")
    return result


def _optional_nonnegative_integer(value: Any, label: str) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        _expect(value.isdigit(), f"{label} must be an integer or blank")
        value = int(value)
    return _integer(value, label, minimum=0)


def _finite_float_text(value: Any, label: str, *, optional: bool = False) -> str:
    number = _number(value, label, optional=optional)
    return "" if number is None else format(number, ".9g")


def _safe_basename(value: Any, label: str) -> str:
    name = _string(value, label)
    normalized = name.replace("\\", "/")
    _expect(normalized == Path(normalized).name, f"{label} must be a basename")
    _expect(name not in (".", ".."), f"{label} is unsafe")
    _expect(not _contains_private_text(name), f"{label} contains private metadata")
    return name


_UUID_RE = re.compile(r"(?i)\b[0-9a-f]{8}-[0-9a-f]{4}-[1-5][0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}\b")
_IPV4_RE = re.compile(r"(?<![0-9.])(?:25[0-5]|2[0-4]\d|1?\d?\d)(?:\.(?:25[0-5]|2[0-4]\d|1?\d?\d)){3}(?![0-9.])")
_EMAIL_RE = re.compile(r"(?i)\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b")
_ABSOLUTE_PATH_RE = re.compile(r"(?:^|[\s='\"])(?:/(?:root|home|inspire|mnt|workspace|tmp|var|etc|opt)/|[A-Za-z]:[\\/])")
_PRIVATE_WORD_RE = re.compile(
    r"(?i)(?:password|passwd|credential|secret|api[_-]?key|access[_-]?token|refresh[_-]?token|authorization|cookie|hostname|user(?:name)?|device[_-]?uuid|job[_-]?id|workspace[_-]?id|project[_-]?id)"
)


def _contains_private_text(value: str) -> bool:
    return bool(
        _UUID_RE.search(value)
        or _IPV4_RE.search(value)
        or _EMAIL_RE.search(value)
        or _ABSOLUTE_PATH_RE.search(value)
        or value.startswith(("/", "file://", "ssh://"))
        or _PRIVATE_WORD_RE.search(value)
    )


def _assert_public(value: Any, location: str = "root") -> None:
    """Reject common credential and internal-infrastructure disclosures."""

    if isinstance(value, dict):
        for key, child in value.items():
            _expect(isinstance(key, str), f"{location} has a non-string key")
            _expect(not _PRIVATE_WORD_RE.search(key), f"private metadata key at {location}.{key}")
            _assert_public(child, f"{location}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _assert_public(child, f"{location}[{index}]")
    elif isinstance(value, str):
        _expect(not _contains_private_text(value), f"private metadata at {location}")
    elif value is not None and type(value) not in (bool, int, float):
        raise SummaryError(f"unsupported public value at {location}: {type(value).__name__}")
    elif isinstance(value, float):
        _expect(math.isfinite(value), f"non-finite public number at {location}")


def _read_json(path: Path) -> dict[str, Any]:
    _expect(path.is_file() and not path.is_symlink(), f"evidence file is missing or symlinked: {path.name}")
    _expect(path.stat().st_size <= MAX_EVIDENCE_JSON_BYTES, f"evidence JSON is unexpectedly large: {path.name}")
    try:
        with path.open("r", encoding="utf-8") as handle:
            value = json.load(handle)
    except (OSError, UnicodeError, json.JSONDecodeError) as error:
        raise SummaryError(f"cannot parse evidence JSON {path.name}: {type(error).__name__}") from error
    return _mapping(value, path.name)


def discover_evidence(raw_root: Path) -> dict[str, list[Evidence]]:
    raw_root = raw_root.expanduser().resolve()
    _expect(raw_root.is_dir(), "--raw-root must be an existing directory")
    _expect(not raw_root.is_symlink(), "--raw-root must not be a symlink")
    found = {BENCHMARK_SCHEMA: [], MIXED_SCHEMA: [], MEMORY_SCHEMA: []}
    for path in sorted(raw_root.rglob("*.json")):
        if path.name.endswith(".trace.json") or path.is_symlink():
            continue
        if path.stat().st_size > MAX_EVIDENCE_JSON_BYTES:
            continue
        try:
            payload = _read_json(path)
        except SummaryError:
            # Non-evidence JSON (for example a suite manifest) is outside this
            # module's schema boundary.  A file advertising a known schema is
            # never ignored because it parses before classification below.
            continue
        schema = payload.get("schema_version")
        if schema in found:
            found[str(schema)].append(Evidence(path=path, payload=payload))
    for schema, records in found.items():
        _expect(records, f"raw root has no {schema} evidence")
    return found


def validate_suite_manifest(raw_root: Path) -> str:
    """Validate suite completion without copying its private path-rich manifest."""

    root = raw_root.expanduser().resolve()
    manifest_path = root / "manifest.json"
    manifest = _read_json(manifest_path)
    _expect(manifest.get("schema_version") == RUN_SUITE_SCHEMA, "raw root has the wrong run-suite manifest schema")
    status = _string(manifest.get("status"), "suite status")
    _expect(status in {"success", "complete_with_oom"}, "suite has not reached an accepted terminal status")
    _expect(manifest.get("dry_run") is False, "suite manifest is a dry-run")
    runtime_root = _string(manifest.get("runtime_root"), "suite runtime root")
    _expect(Path(runtime_root).expanduser().resolve() == root, "suite manifest does not belong to --raw-root")
    expected_matrix = _mapping(manifest.get("expected_matrix"), "suite expected matrix")
    _expect(expected_matrix.get("base_command_count") == 23, "suite expected matrix does not contain the 23 fixed base cases")
    gpu = _mapping(manifest.get("gpu"), "suite GPU contract")
    _expect(gpu.get("expected_count") == 8 and gpu.get("visible_count") == 8, "suite was not executed with the fixed eight-GPU contract")
    _expect(gpu.get("selectors_recorded") is False, "suite manifest recorded private GPU selectors")
    execution = _mapping(manifest.get("execution_contract"), "suite execution contract")
    for field in ("single_node", "one_process_per_gpu", "each_command_launched_once", "case_local_home_tmp_and_caches"):
        _expect(execution.get(field) is True, f"suite execution contract does not guarantee {field}")
    for field in ("automatic_retry", "credentials_inherited", "conda_and_user_python_state_inherited", "formal_child_commands_use_dry_run"):
        _expect(execution.get(field) is False, f"suite execution contract violates {field}")

    stages = [_mapping(value, "suite stage") for value in _sequence(manifest.get("stages"), "suite stages")]
    expected_stage_counts = {"benchmark": 4, "torch_profile": 6, "mixed_precision": 5, "memory_snapshot": 8}
    _expect({stage.get("name") for stage in stages} == set(expected_stage_counts), "suite stage set is incomplete")
    commands: list[dict[str, Any]] = []
    for stage in stages:
        name = _string(stage.get("name"), "suite stage name")
        stage_status = _string(stage.get("status"), f"suite {name} status")
        _expect(stage_status in {"success", "complete_with_oom"}, f"suite stage {name} is not complete")
        stage_commands = [_mapping(value, f"suite {name} command") for value in _sequence(stage.get("commands"), f"suite {name} commands")]
        _expect(len(stage_commands) >= expected_stage_counts[name], f"suite stage {name} has too few commands")
        if name != "memory_snapshot":
            _expect(len(stage_commands) == expected_stage_counts[name], f"suite stage {name} has unexpected extra commands")
        for command in stage_commands:
            command_status = _string(command.get("status"), f"suite {name} command status")
            _expect(command_status in {"success", "oom"}, f"suite {name} contains a hard-failed command")
            _expect(command.get("attempts") == 1 and command.get("automatic_retry") is False, f"suite {name} command was retried")
            _expect(command.get("returncode") == 0, f"suite {name} command did not exit cleanly")
            expected_reported = "ok" if command_status == "success" else "oom"
            _expect(command.get("reported_status") == expected_reported, f"suite {name} command status disagrees with its artifact")
            commands.append(command)
    identifiers = [_string(command.get("case_id"), "suite case id") for command in commands]
    _expect(len(identifiers) == len(set(identifiers)), "suite manifest contains duplicate case identifiers")
    oom_count = sum(command.get("status") == "oom" for command in commands)
    expected_status = "complete_with_oom" if oom_count else "success"
    _expect(status == expected_status, "suite terminal status disagrees with nested command statuses")
    summary = _mapping(manifest.get("summary"), "suite summary")
    _expect(summary.get("command_count") == len(commands), "suite summary command count is wrong")
    _expect(summary.get("oom_count") == oom_count and summary.get("failure_count") == 0, "suite summary failure/OOM counts are wrong")

    completion_record = _mapping(manifest.get("completion_marker"), "suite completion marker record")
    _expect(completion_record.get("created") is True, "suite completion marker was not created")
    completion_relative = _string(completion_record.get("path"), "suite completion marker path")
    _expect(completion_relative == "markers/suite.complete.json", "suite completion marker path is unexpected")
    completion = _read_json(root / completion_relative)
    _expect(completion.get("schema_version") == RUN_SUITE_SCHEMA and completion.get("status") == status, "suite completion marker disagrees with the manifest")
    _expect(completion.get("manifest") == "manifest.json", "suite completion marker points at the wrong manifest")
    _expect(completion.get("expected_base_command_count") == 23 and completion.get("actual_command_count") == len(commands), "suite completion marker counts are wrong")
    return status


def _check_authoritative(evidence: Evidence, schema: str, *, statuses: tuple[str, ...] = ("ok",)) -> None:
    payload = evidence.payload
    _expect(payload.get("schema_version") == schema, f"wrong schema in {evidence.path.name}")
    _expect(payload.get("status") in statuses, f"non-success status in {evidence.path.name}")
    _expect(payload.get("authoritative") is True, f"non-authoritative evidence in {evidence.path.name}")


def _recomputed_statistics(raw: Sequence[float]) -> tuple[float, float, float]:
    _expect(len(raw) >= 2, "sample statistics require at least two values")
    mean = statistics.fmean(raw)
    sample_std = statistics.stdev(raw)
    return mean, sample_std, sample_std / mean


def _close(actual: Any, expected: float, label: str) -> None:
    number = _number(actual, label)
    assert number is not None
    tolerance = max(1e-9, abs(expected) * 1e-8)
    _expect(abs(number - expected) <= tolerance, f"{label} disagrees with raw samples")


def _environment_from_benchmark(payload: dict[str, Any]) -> dict[str, Any]:
    metadata = _mapping(payload.get("metadata"), "benchmark.metadata")
    _expect(metadata.get("starter_commit") == STARTER_COMMIT, "benchmark uses the wrong starter commit")
    hardware = _mapping(metadata.get("hardware"), "benchmark.metadata.hardware")
    software = _mapping(metadata.get("software"), "benchmark.metadata.software")
    _expect(hardware.get("device_type") == "cuda", "benchmark metadata is not CUDA")
    _expect(_mapping(metadata.get("privacy"), "benchmark.metadata.privacy").get("public_allowlist") is True, "benchmark metadata was not produced with the public allowlist")
    gpu = _mapping(hardware.get("gpu"), "benchmark.metadata.hardware.gpu")
    result = {
        "gpu_model": _string(gpu.get("model"), "gpu model"),
        "gpu_total_memory_bytes": _integer(gpu.get("total_memory_bytes"), "GPU memory", minimum=1),
        "compute_capability": [_integer(item, "compute capability", minimum=0) for item in _sequence(gpu.get("compute_capability"), "compute capability")],
        "cuda_driver_version": hardware.get("cuda_driver_version"),
        "python_version": _string(software.get("python_version"), "Python version"),
        "pytorch_version": _string(software.get("torch_version"), "PyTorch version"),
        "cuda_runtime_version": _string(software.get("cuda_runtime_version"), "CUDA runtime version"),
        "cudnn_version": software.get("cudnn_version"),
    }
    _assert_public(result, "environment")
    return result


def _validate_benchmark_contract(payload: dict[str, Any], label: str) -> None:
    contract = _mapping(payload.get("measurement_contract"), f"{label}.measurement_contract")
    _expect(contract.get("clock") == "time.perf_counter", f"{label} uses the wrong timer")
    _expect(
        contract.get("cuda_synchronize_at_step_boundaries") is True,
        f"{label} lacks CUDA synchronization at step boundaries",
    )
    _expect(
        contract.get("cuda_synchronize_after_each_phase") is False,
        f"{label} uses phase-level synchronization that distorts end-to-end timing",
    )
    _expect(contract.get("initialization_and_data_generation_timed") is False, f"{label} includes initialization/data generation")
    _expect(contract.get("forward_uses_inference_mode") is True, f"{label} forward is not inference-only")
    _expect(contract.get("forward_backward_clears_gradients_each_step") is True, f"{label} accumulates gradients across steps")
    _expect(
        contract.get("train_step_total_includes") == ["zero_grad", "forward", "loss", "backward", "optimizer"],
        f"{label} train_step timing boundary is incomplete",
    )


def _validate_measurement(result: dict[str, Any], expected_steps: int, label: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    _expect(result.get("status") == "ok", f"{label} result is not ok")
    raw_steps = [_mapping(item, f"{label}.raw_steps") for item in _sequence(result.get("raw_steps"), f"{label}.raw_steps")]
    _expect(len(raw_steps) == expected_steps, f"{label} has the wrong raw sample count")
    raw_totals = []
    for index, step in enumerate(raw_steps, start=1):
        _expect(_integer(step.get("step"), f"{label}.step") == index, f"{label} step numbering is not contiguous")
        total = _number(step.get("total_ms"), f"{label}.total_ms", positive=True)
        assert total is not None
        raw_totals.append(total)
        phases = _mapping(step.get("phases_ms"), f"{label}.phases_ms")
        for phase, duration in phases.items():
            _expect(phase in {"zero_grad", "forward", "loss", "backward", "optimizer"}, f"{label} has an unknown phase")
            _number(duration, f"{label}.{phase}", positive=True)
        loss = step.get("loss")
        if loss is not None:
            _number(loss, f"{label}.loss")
    statistics_payload = _mapping(result.get("total_statistics"), f"{label}.total_statistics")
    stored_raw = [_number(item, f"{label}.total_statistics.raw_ms", positive=True) for item in _sequence(statistics_payload.get("raw_ms"), f"{label}.raw_ms")]
    _expect(stored_raw == raw_totals, f"{label} total raw samples disagree")
    if len(raw_totals) == 1:
        mean = raw_totals[0]
        _close(statistics_payload.get("mean_ms"), mean, f"{label}.mean_ms")
        _expect(statistics_payload.get("sample_std_ms") is None and statistics_payload.get("cv") is None, f"{label} one-sample statistics must leave sample std/CV null")
    else:
        mean, sample_std, cv = _recomputed_statistics(raw_totals)
        _close(statistics_payload.get("mean_ms"), mean, f"{label}.mean_ms")
        _close(statistics_payload.get("sample_std_ms"), sample_std, f"{label}.sample_std_ms")
        _close(statistics_payload.get("cv"), cv, f"{label}.cv")
    return raw_steps, statistics_payload


def build_benchmark_rows(records: Sequence[Evidence]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    indexed: dict[tuple[int, str], tuple[dict[str, Any], dict[str, Any]]] = {}
    environment: dict[str, Any] | None = None
    for evidence in records:
        _check_authoritative(evidence, BENCHMARK_SCHEMA)
        payload = evidence.payload
        _expect(payload.get("profile") is None, f"profile run cannot enter benchmark.csv: {evidence.path.name}")
        config = _mapping(payload.get("configuration"), "benchmark.configuration")
        _validate_benchmark_contract(payload, "benchmark")
        _expect(config.get("requested_model_size") == "small" and config.get("effective_model_size") == "small", "benchmark must use small")
        _expect(config.get("requested_batch_size") == 4 and config.get("batch_size") == 4, "benchmark must use batch 4")
        _expect(config.get("requested_context_length") == 512 and config.get("context_length") == 512, "benchmark must use context 512")
        for key, expected in MODEL_DIMENSIONS["small"].items():
            _expect(config.get(key) == expected, f"benchmark small model has wrong {key}")
        _expect(config.get("dtype") == "fp32" and config.get("device") == "cuda", "benchmark must be CUDA FP32")
        _expect(config.get("seed") == 42, "benchmark seed must be 42")
        warmup = _integer(config.get("warmup_steps"), "benchmark warmup", minimum=0)
        _integer(config.get("measurement_steps"), "benchmark steps", minimum=10)
        modes = _sequence(config.get("modes"), "benchmark modes")
        allowed_modes = {"forward", "forward_backward", "train_step"} if warmup == 5 else {"train_step"} if warmup == 0 else set()
        expected_modes = set(modes)
        _expect(
            expected_modes and expected_modes.issubset(allowed_modes) and len(modes) == len(expected_modes),
            "benchmark requires warmup=5 cases for three modes and warmup=0 for train_step",
        )
        results = [_mapping(item, "benchmark.results") for item in _sequence(payload.get("results"), "benchmark.results")]
        _expect(len(results) == len(expected_modes), "benchmark result count disagrees with modes")
        for result in results:
            mode = _string(result.get("mode"), "benchmark mode")
            _expect(mode in expected_modes, "unexpected benchmark mode")
            key = (warmup, mode)
            _expect(key not in indexed, f"duplicate benchmark case: {key}")
            indexed[key] = (config, result)
        candidate_environment = _environment_from_benchmark(payload)
        if environment is None:
            environment = candidate_environment
        else:
            _expect(environment == candidate_environment, "benchmark runs used inconsistent environments")

    expected_keys = {(5, mode) for mode in ("forward", "forward_backward", "train_step")} | {(0, "train_step")}
    _expect(set(indexed) == expected_keys, "benchmark matrix is incomplete or contains extra cases")
    rows: list[dict[str, Any]] = []
    mode_order = {"forward": 0, "forward_backward": 1, "train_step": 2}
    for warmup, mode in sorted(indexed, key=lambda item: (-item[0], mode_order[item[1]])):
        config, result = indexed[(warmup, mode)]
        expected_steps = _integer(config.get("measurement_steps"), "benchmark steps", minimum=10)
        raw_steps, stats = _validate_measurement(result, expected_steps, f"benchmark {warmup}/{mode}")
        memory = _mapping(result.get("memory"), "benchmark memory")
        run_id = f"small_b4_ctx512_fp32_{mode}_warmup{warmup}"
        for step in raw_steps:
            phases = _mapping(step.get("phases_ms"), "benchmark phases")
            row = {
                "run_id": run_id,
                "model_size": "small",
                "batch_size": 4,
                "context_length": 512,
                "dtype": "fp32",
                "mode": mode,
                "warmup_steps": warmup,
                "measurement_steps": expected_steps,
                "measurement_step": step["step"],
                "total_ms": _finite_float_text(step["total_ms"], "total_ms"),
                "zero_grad_ms": _finite_float_text(phases.get("zero_grad"), "zero_grad", optional=True),
                "forward_ms": _finite_float_text(phases.get("forward"), "forward", optional=True),
                "loss_ms": _finite_float_text(phases.get("loss"), "loss", optional=True),
                "backward_ms": _finite_float_text(phases.get("backward"), "backward", optional=True),
                "optimizer_ms": _finite_float_text(phases.get("optimizer"), "optimizer", optional=True),
                "loss": _finite_float_text(step.get("loss"), "loss", optional=True),
                "mean_ms": _finite_float_text(stats.get("mean_ms"), "mean"),
                "sample_std_ms": _finite_float_text(stats.get("sample_std_ms"), "std"),
                "cv": _finite_float_text(stats.get("cv"), "cv"),
                "cv_percent": _finite_float_text(stats.get("cv_percent"), "cv percent"),
                "peak_allocated_bytes": _optional_nonnegative_integer(memory.get("peak_allocated_bytes"), "peak allocated"),
                "peak_active_bytes": _optional_nonnegative_integer(memory.get("peak_active_bytes"), "peak active"),
                "peak_reserved_bytes": _optional_nonnegative_integer(memory.get("peak_reserved_bytes"), "peak reserved"),
            }
            _assert_public(row, "benchmark row")
            rows.append(row)
    assert environment is not None
    return rows, environment


def _read_profile_csv(path: Path) -> list[dict[str, Any]]:
    _expect(path.is_file() and not path.is_symlink(), f"missing profile summary: {path.name}")
    _expect(path.stat().st_size <= 16 * 1024 * 1024, f"profile summary is unexpectedly large: {path.name}")
    try:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            _expect(reader.fieldnames == ["scope", "op_or_kernel", "name", "calls", "CPU_us", "CUDA_us", "attribution"], f"wrong profile CSV schema: {path.name}")
            source = list(reader)
    except (OSError, UnicodeError, csv.Error) as error:
        raise SummaryError(f"cannot read profile summary {path.name}: {type(error).__name__}") from error
    _expect(source, f"empty profile summary: {path.name}")
    result: list[dict[str, Any]] = []
    for index, row in enumerate(source, start=2):
        name = _string(row.get("name"), f"{path.name}:{index} name")
        _expect(not _contains_private_text(name), f"private operator name in {path.name}:{index}")
        calls = _optional_nonnegative_integer(row.get("calls"), f"{path.name}:{index} calls")
        _expect(calls is not None and calls > 0, f"Calls must be positive in {path.name}:{index}")
        cpu = _number(row.get("CPU_us"), f"{path.name}:{index} CPU_us", optional=True)
        cuda = _number(row.get("CUDA_us"), f"{path.name}:{index} CUDA_us", optional=True)
        _expect(cpu is None or cpu >= 0.0, f"negative CPU time in {path.name}:{index}")
        _expect(cuda is None or cuda >= 0.0, f"negative CUDA time in {path.name}:{index}")
        scope = _string(row.get("scope"), f"{path.name}:{index} scope")
        row_type = _string(row.get("op_or_kernel"), f"{path.name}:{index} type")
        _expect(scope in {"profiler_native", "synchronized_phase_wall"}, f"unknown scope in {path.name}:{index}")
        _expect(row_type in {"annotation", "operator", "phase_wall"}, f"unknown row type in {path.name}:{index}")
        result.append(
            {
                "scope": scope,
                "op_or_kernel": row_type,
                "name": name,
                "calls": calls,
                "cpu_time_us": cpu,
                "cuda_time_us": cuda,
            }
        )
    return result


def _canonical_annotation_rows(
    rows: Sequence[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Keep one CPU-side aggregate for each logical profiler range.

    A CUDA trace can contain a GPU-side mirror of the same named user range.
    The CPU-side ``record_function`` aggregate is the one public source of
    Calls and correlated CUDA time; exposing both would invite double-counting.
    """

    canonical: list[dict[str, Any]] = []
    mirrors: list[dict[str, Any]] = []
    for name in REQUIRED_RANGES:
        candidates = [
            row
            for row in rows
            if row["scope"] == "profiler_native"
            and row["op_or_kernel"] == "annotation"
            and row["name"] == name
        ]
        _expect(candidates, f"profile range lacks a native annotation aggregate: {name}")
        cpu_side = [
            row
            for row in candidates
            if row["cpu_time_us"] is not None and float(row["cpu_time_us"]) > 0.0
        ]
        _expect(
            len(cpu_side) == 1,
            f"profile range needs exactly one CPU-side annotation aggregate: {name}",
        )
        selected = cpu_side[0]
        canonical.append(selected)
        mirrors.extend(row for row in candidates if row is not selected)
    return canonical, mirrors


def _logical_benchmark_command(model: str, context: int, result_name: str) -> list[str]:
    return [
        "python",
        "profiling/benchmark.py",
        "--model-size",
        model,
        "--batch-size",
        "4",
        "--context-length",
        str(context),
        "--mode",
        "train_step",
        "--warmup",
        "5",
        "--steps",
        "1",
        "--dtype",
        "fp32",
        "--seed",
        "42",
        "--device",
        "cuda",
        "--profile",
        "torch",
        "--output",
        f"results/profile/{result_name}",
    ]


def build_profile_outputs(records: Sequence[Evidence]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    indexed: dict[tuple[str, int], tuple[Evidence, dict[str, Any], dict[str, Any], list[dict[str, Any]]]] = {}
    environment: dict[str, Any] | None = None
    for evidence in records:
        _check_authoritative(evidence, BENCHMARK_SCHEMA)
        payload = evidence.payload
        profile = payload.get("profile")
        _expect(isinstance(profile, dict), f"non-profile benchmark mixed into profile matrix: {evidence.path.name}")
        _validate_benchmark_contract(payload, "profile")
        config = _mapping(payload.get("configuration"), "profile.configuration")
        model = _string(config.get("requested_model_size"), "profile model")
        context = _integer(config.get("context_length"), "profile context", minimum=1)
        _expect(model in PROFILE_MODELS and context in PROFILE_CONTEXTS, "profile configuration is outside the fixed matrix")
        _expect(config.get("effective_model_size") == model, "profile model was silently reduced")
        _expect(config.get("requested_batch_size") == 4 and config.get("batch_size") == 4, "profile must use batch 4")
        _expect(config.get("requested_context_length") == context and config.get("context_length") == context, "profile requested/effective context differs")
        _expect(config.get("dtype") == "fp32" and config.get("device") == "cuda", "profile must use CUDA FP32")
        for dimension, expected_value in MODEL_DIMENSIONS[model].items():
            _expect(config.get(dimension) == expected_value, f"profile {model} has wrong {dimension}")
        _expect(config.get("warmup_steps") >= 5 and config.get("measurement_steps") == 1, "profile must capture one step after five warmups")
        _expect(config.get("modes") == ["train_step"], "profile must use train_step")
        _expect(config.get("seed") == 42, "profile seed must be 42")
        _expect(profile.get("tool") == "torch.profiler", "all profiles must use torch.profiler")
        activities = set(_sequence(profile.get("activities"), "profile activities"))
        _expect({"CPU", "CUDA"}.issubset(activities), "profile requires CPU and CUDA activities")
        _expect(
            profile.get("measured_steps") == 1
            and profile.get("warmup_steps_before_measurement") >= 5
            and profile.get("warmup_steps_in_trace") == 0,
            "profile measurement boundary is invalid",
        )
        _expect(profile.get("native_cuda_attribution") is True, "profile lacks native CUDA attribution")
        trace_basename = _safe_basename(profile.get("trace_file"), "profile trace filename")
        summary_basename = _safe_basename(profile.get("summary_file"), "profile summary filename")
        trace_path = evidence.path.parent / trace_basename
        summary_path = evidence.path.parent / summary_basename
        _expect(trace_path.is_file() and not trace_path.is_symlink(), f"local Chrome trace is missing: {trace_basename}")
        source_rows = _read_profile_csv(summary_path)
        names = {row["name"] for row in source_rows}
        missing_ranges = set(REQUIRED_RANGES) - names
        _expect(not missing_ranges, f"profile {model}/{context} is missing ranges: {sorted(missing_ranges)}")
        _expect(any((row["cuda_time_us"] or 0.0) > 0.0 for row in source_rows), f"profile {model}/{context} has no CUDA time")
        results = [_mapping(item, "profile results") for item in _sequence(payload.get("results"), "profile results")]
        _expect(len(results) == 1 and results[0].get("mode") == "train_step", "profile must contain one train_step result")
        _validate_measurement(results[0], 1, f"profile {model}/{context}")
        key = (model, context)
        _expect(key not in indexed, f"duplicate profile case: {key}")
        indexed[key] = (evidence, profile, results[0], source_rows)
        candidate_environment = _environment_from_benchmark(payload)
        if environment is None:
            environment = candidate_environment
        else:
            _expect(environment == candidate_environment, "profile runs used inconsistent environments")

    expected = {(model, context) for model in PROFILE_MODELS for context in PROFILE_CONTEXTS}
    _expect(set(indexed) == expected, "the six-run profile matrix is incomplete or has extras")
    public_rows: list[dict[str, Any]] = []
    metadata_runs: list[dict[str, Any]] = []
    representative: dict[str, Any] | None = None
    for model, context in sorted(indexed, key=lambda key: (PROFILE_MODELS.index(key[0]), key[1])):
        evidence, profile, result, source_rows = indexed[(model, context)]
        trace_basename = _safe_basename(profile["trace_file"], "profile trace filename")
        trace_path = evidence.path.parent / trace_basename
        kernel_rows = extract_profile_kernel_aggregates(trace_path)
        canonical_annotations, annotation_mirrors = _canonical_annotation_rows(source_rows)
        phase_wall_rows = [row for row in source_rows if row["scope"] == "synchronized_phase_wall"]
        required = canonical_annotations + phase_wall_rows
        excluded_row_ids = {id(row) for row in required + annotation_mirrors}
        native = sorted(
            (row for row in source_rows if id(row) not in excluded_row_ids),
            key=lambda row: (-(row["cuda_time_us"] or 0.0), -(row["cpu_time_us"] or 0.0), row["name"]),
        )[:40]
        selected: list[dict[str, Any]] = []
        seen: set[tuple[Any, ...]] = set()
        for row in sorted(required, key=lambda item: (item["scope"], item["name"])) + native + kernel_rows:
            identity = (row["scope"], row["op_or_kernel"], row["name"], row["calls"], row["cpu_time_us"], row["cuda_time_us"])
            if identity not in seen:
                seen.add(identity)
                selected.append(row)
        run_id = f"{model}_b4_ctx{context}_fp32_train_step"
        result_basename = _safe_basename(evidence.path.name, "profile result filename")
        command = _logical_benchmark_command(model, context, result_basename)
        command_text = " ".join(command)
        for row in selected:
            if row["scope"] == "synchronized_phase_wall":
                attribution = "synchronized wall clock; CUDA time intentionally blank"
                evidence_source = "synchronized_wall_clock"
            elif row["scope"] == "profiler_trace_kernel":
                attribution = "local Chrome trace cat=kernel aggregate inside profile/measure"
                evidence_source = "torch.profiler.chrome_trace_kernel"
            elif row["op_or_kernel"] == "annotation":
                attribution = "CPU-side torch.profiler record_function aggregate; paired GPU annotation mirror omitted"
                evidence_source = "torch.profiler.record_function_cpu_annotation"
            else:
                attribution = "torch.profiler native aggregate"
                evidence_source = "torch.profiler.key_averages"
            public_row = {
                "run_id": run_id,
                "model_size": model,
                "batch_size": 4,
                "context_length": context,
                "mode": "train_step",
                "dtype": "fp32",
                "tool": "torch.profiler",
                "command": command_text,
                "scope": row["scope"],
                "op_or_kernel": row["op_or_kernel"],
                "evidence_source": evidence_source,
                "name": row["name"],
                "calls": row["calls"],
                "cpu_time_us": "" if row["cpu_time_us"] is None else format(row["cpu_time_us"], ".9g"),
                "cuda_time_us": "" if row["cuda_time_us"] is None else format(row["cuda_time_us"], ".9g"),
                "attribution": attribution,
            }
            _assert_public(public_row, "profile row")
            public_rows.append(public_row)
        metadata_runs.append(
            {
                "run_id": run_id,
                "model_size": model,
                "batch_size": 4,
                "context_length": context,
                "mode": "train_step",
                "dtype": "fp32",
                "warmup_steps": 5,
                "warmup_steps_in_trace": 0,
                "profile_warmup_marker": "post_warmup_synchronization_only",
                "measurement_steps": 1,
                "tool": "torch.profiler",
                "activities": ["CPU", "CUDA"],
                "command": command,
                "local_trace_basename": trace_basename,
                "local_trace_retained": True,
                "full_trace_published": False,
                "local_trace_relation": "isolated per-run work directory keyed by run_id; basename is not globally unique",
                "public_annotation_row_count": len(canonical_annotations),
                "omitted_annotation_mirror_row_count": len(annotation_mirrors),
                "public_summary_row_count": len(selected),
                "public_trace_kernel_row_count": len(kernel_rows),
                "published_trace_kernel_calls": sum(int(row["calls"]) for row in kernel_rows),
                "published_trace_kernel_cuda_time_us": sum(float(row["cuda_time_us"]) for row in kernel_rows),
            }
        )
        if (model, context) == ("xl", 1_024):
            representative = {
                "run_id": run_id,
                "rows": selected,
                "result": result,
                "trace_path": trace_path,
            }
    assert environment is not None and representative is not None
    metadata = {
        "schema_version": PUBLIC_SCHEMA,
        "assignment": "A2-P",
        "handout_version": HANDOUT_VERSION,
        "starter_commit": STARTER_COMMIT,
        "tool": "torch.profiler",
        "tool_limit": "framework operator and CUDA activity view; no nsys-specific CUDA API correlation is claimed",
        "required_ranges": list(REQUIRED_RANGES),
        "matrix": {"models": list(PROFILE_MODELS), "contexts": list(PROFILE_CONTEXTS), "mode": "train_step", "dtype": "fp32"},
        "environment": environment,
        "runs": metadata_runs,
        "representative_run_id": representative["run_id"],
        "representative_asset": "assets/compute_profile.png",
        "annotation_projection": {
            "scope": "profiler_native annotation",
            "source": "CPU-side torch.profiler record_function aggregate",
            "selection": "one positive-CPU row per required logical range; paired GPU annotation mirrors are omitted",
            "calls_and_cuda_time_policy": "do not add rows with the same range name",
        },
        "trace_kernel_projection": {
            "source_category": "kernel",
            "scope": "profiler_trace_kernel",
            "aggregation": "exact name, calls and summed source duration_us inside profile/measure",
            "maximum_rows_per_run": MAX_PROFILE_KERNEL_ROWS_PER_RUN,
            "cpu_time_policy": "blank because these rows are CUDA kernel events",
        },
        "raw_artifact_policy": {"chrome_traces_retained_locally": True, "chrome_traces_published": False},
    }
    _assert_public(metadata, "profile metadata")
    return public_rows, metadata, representative


def _normalize_accumulation(payload: dict[str, Any]) -> dict[str, Any]:
    accumulation = _mapping(payload.get("accumulation"), "mixed.accumulation")
    records = []
    for record in _sequence(accumulation.get("records"), "mixed accumulation records"):
        item = _mapping(record, "mixed accumulation record")
        records.append(
            {
                "case": _string(item.get("case"), "accumulation case"),
                "input_dtype": _string(item.get("input_dtype"), "accumulation input dtype"),
                "accumulator_dtype": _string(item.get("accumulator_dtype"), "accumulation accumulator dtype"),
                "iterations": _integer(item.get("iterations"), "accumulation iterations", minimum=1),
                "increment": _number(item.get("increment"), "accumulation increment"),
                "actual_value": _number(item.get("actual_value"), "accumulation actual"),
                "mathematical_value": _number(item.get("mathematical_value"), "accumulation mathematical"),
                "absolute_error": _number(item.get("absolute_error"), "accumulation error"),
            }
        )
    _expect(len(records) == 4 and len({item["case"] for item in records}) == 4, "mixed precision requires exactly four accumulation cases")
    expected_cases = {
        "fp32_value_fp32_accumulator",
        "fp16_value_fp16_accumulator",
        "fp16_value_fp32_accumulator_implicit_cast",
        "fp16_value_fp32_accumulator_explicit_cast",
    }
    _expect({item["case"] for item in records} == expected_cases, "mixed precision accumulation case labels differ from the pinned PDF")
    for item in records:
        mathematical = float(item["iterations"]) * float(item["increment"])
        _close(item["mathematical_value"], mathematical, f"accumulation {item['case']} mathematical value")
        _close(item["absolute_error"], abs(float(item["actual_value"]) - mathematical), f"accumulation {item['case']} absolute error")
    result = {
        "source": "pinned assignment PDF",
        "starter_commit": STARTER_COMMIT,
        "executed_as_written": accumulation.get("executed_as_written") is True,
        "records": records,
    }
    _expect(result["executed_as_written"], "accumulation snippets were not marked as executed as written")
    return result


def _normalize_toy(payload: dict[str, Any]) -> dict[str, Any]:
    toy = _mapping(payload.get("toy_model"), "mixed.toy_model")
    _expect(toy.get("status") == "ok" and toy.get("authoritative_cuda_bf16") is True, "ToyModel must be authoritative CUDA BF16")
    scalar_fields = (
        "device_type",
        "autocast_enabled",
        "autocast_dtype",
        "fc1_output_dtype",
        "layer_norm_output_dtype",
        "logits_dtype",
        "loss_dtype",
        "loss_value",
    )
    result = {field: toy.get(field) for field in scalar_fields}
    result.update(
        {
            "status": "ok",
            "authoritative_cuda_bf16": True,
            "parameter_dtypes": _mapping(toy.get("parameter_dtypes"), "ToyModel parameter dtypes"),
            "parameter_dtype_set": _sequence(toy.get("parameter_dtype_set"), "ToyModel parameter dtype set"),
            "gradient_dtypes": _mapping(toy.get("gradient_dtypes"), "ToyModel gradient dtypes"),
            "gradient_dtype_set": _sequence(toy.get("gradient_dtype_set"), "ToyModel gradient dtype set"),
        }
    )
    _expect(result["device_type"] == "cuda" and result["autocast_dtype"] == "bfloat16", "ToyModel must use CUDA BF16 autocast")
    for field in ("fc1_output_dtype", "layer_norm_output_dtype", "logits_dtype", "loss_dtype"):
        _string(result[field], f"ToyModel {field}")
    _number(result["loss_value"], "ToyModel loss")
    _expect(result["parameter_dtypes"] and result["gradient_dtypes"], "ToyModel parameter/gradient dtype maps are empty")
    _assert_public(result, "ToyModel")
    return result


def _normalize_precision_record(record: Any, precision: str, expected_config: dict[str, Any]) -> dict[str, Any]:
    item = _mapping(record, f"mixed {precision} record")
    status = _string(item.get("status"), f"mixed {precision} status")
    _expect(status in {"ok", "oom"} and item.get("precision") == precision, f"mixed {precision} record is invalid")
    _expect(_mapping(item.get("configuration"), f"mixed {precision} configuration") == expected_config, "FP32/BF16 configurations differ")
    _expect(item.get("measurement_mode") == "forward_backward", f"mixed {precision} must measure forward+loss+backward")
    _expect(item.get("optimizer_step_included") is False, f"mixed {precision} must not include an optimizer step")
    memory = _mapping(item.get("memory"), f"mixed {precision} memory")
    _expect(memory.get("authoritative") is True, f"mixed {precision} memory is non-authoritative")
    normalized_memory = {key: _integer(memory.get(key), f"mixed {precision} {key}", minimum=0) for key in ("peak_allocated_bytes", "peak_reserved_bytes", "peak_active_bytes")}
    if status == "oom":
        result = {
            "status": "oom",
            "precision": precision,
            "measurement_mode": "forward_backward",
            "measurement_boundary": ["forward", "loss", "backward"],
            "optimizer_step_included": False,
            "failure": {
                "type": _string(item.get("error_type"), f"mixed {precision} OOM type"),
                "stage": "language_model_precision_run",
            },
            "memory": normalized_memory,
        }
        _assert_public(result, f"mixed {precision} OOM")
        return result
    timing = _mapping(item.get("timing"), f"mixed {precision} timing")
    raw = [_number(value, f"mixed {precision} raw timing", positive=True) for value in _sequence(timing.get("raw_ms"), f"mixed {precision} raw timing")]
    expected_steps = _integer(expected_config.get("measurement_steps"), "mixed measurement steps", minimum=10)
    _expect(len(raw) == expected_steps, f"mixed {precision} timing sample count is wrong")
    mean, sample_std, cv = _recomputed_statistics([float(item) for item in raw])
    _close(timing.get("mean_ms"), mean, f"mixed {precision} mean")
    _close(timing.get("sample_std_ms"), sample_std, f"mixed {precision} std")
    _close(timing.get("cv"), cv, f"mixed {precision} cv")
    phase_timings = _mapping(item.get("phase_timings"), f"mixed {precision} phase timings")
    normalized_phases: dict[str, Any] = {}
    phase_raw: dict[str, list[float]] = {}
    for phase in ("forward_including_loss", "backward"):
        phase_payload = _mapping(phase_timings.get(phase), f"mixed {precision} {phase}")
        values = [
            float(_number(value, f"mixed {precision} {phase} raw timing", positive=True))
            for value in _sequence(phase_payload.get("raw_ms"), f"mixed {precision} {phase} raw timing")
        ]
        _expect(len(values) == expected_steps, f"mixed {precision} {phase} sample count is wrong")
        phase_mean, phase_std, phase_cv = _recomputed_statistics(values)
        _close(phase_payload.get("mean_ms"), phase_mean, f"mixed {precision} {phase} mean")
        _close(phase_payload.get("sample_std_ms"), phase_std, f"mixed {precision} {phase} std")
        _close(phase_payload.get("cv"), phase_cv, f"mixed {precision} {phase} cv")
        phase_raw[phase] = values
        normalized_phases[phase] = {
            "raw_ms": values,
            "sample_count": len(values),
            "mean_ms": phase_mean,
            "sample_std_ms": phase_std,
            "cv": phase_cv,
            "cv_percent": phase_cv * 100.0,
            "min_ms": min(values),
            "max_ms": max(values),
        }
    for index, total in enumerate(raw):
        _close(total, phase_raw["forward_including_loss"][index] + phase_raw["backward"][index], f"mixed {precision} total sample {index + 1}")
    loss = _mapping(item.get("loss"), f"mixed {precision} loss")
    losses = [_number(value, f"mixed {precision} loss") for value in _sequence(loss.get("raw"), f"mixed {precision} losses")]
    _expect(len(losses) == expected_steps and loss.get("all_finite") is True, f"mixed {precision} loss trend is invalid")
    logits = _mapping(item.get("final_logits"), f"mixed {precision} logits")
    _expect(logits.get("all_finite") is True, f"mixed {precision} logits are non-finite")
    result = {
        "status": "ok",
        "precision": precision,
        "autocast": item.get("autocast"),
        "autocast_dtype": item.get("autocast_dtype"),
        "model_parameter_dtype_set": _sequence(item.get("model_parameter_dtype_set"), f"mixed {precision} parameter dtypes"),
        "measurement_mode": "forward_backward",
        "measurement_boundary": ["forward", "loss", "backward"],
        "optimizer_step_included": False,
        "timing": {
            "clock": "time.perf_counter",
            "unit": "milliseconds",
            "cuda_synchronize_before_and_after_each_measurement": timing.get("cuda_synchronize_before_and_after_each_measurement") is True,
            "data_generation_and_initialization_timed": timing.get("data_generation_and_initialization_timed") is True,
            "gradient_clearing_timed": timing.get("gradient_clearing_timed") is True,
            "raw_ms": raw,
            "sample_count": len(raw),
            "mean_ms": mean,
            "sample_std_ms": sample_std,
            "cv": cv,
            "cv_percent": cv * 100.0,
            "min_ms": min(raw),
            "max_ms": max(raw),
        },
        "phase_timings": normalized_phases,
        "memory": normalized_memory,
        "loss": {"raw": losses, "first": losses[0], "last": losses[-1], "change": losses[-1] - losses[0], "all_finite": True},
        "final_logits": {
            "dtype": _string(logits.get("dtype"), "logits dtype"),
            "mean": _number(logits.get("mean"), "logits mean"),
            "std": _number(logits.get("std"), "logits std"),
            "l2_norm": _number(logits.get("l2_norm"), "logits norm"),
            "all_finite": True,
        },
    }
    _expect(result["timing"]["cuda_synchronize_before_and_after_each_measurement"], "mixed timing lacks CUDA synchronization")
    _expect(not result["timing"]["data_generation_and_initialization_timed"], "mixed timing includes initialization")
    _expect(not result["timing"]["gradient_clearing_timed"], "mixed timing includes gradient clearing")
    return result


def _logical_mixed_command(model: str, seed: int) -> list[str]:
    return [
        "python",
        "profiling/mixed_precision.py",
        "--device",
        "cuda",
        "--model-size",
        model,
        "--seed",
        str(seed),
        "--output",
        f"results/mixed/{model}.json",
    ]


def build_mixed_payload(records: Sequence[Evidence]) -> dict[str, Any]:
    indexed: dict[str, dict[str, Any]] = {}
    shared_accumulation: dict[str, Any] | None = None
    shared_toy: dict[str, Any] | None = None
    environment: dict[str, Any] | None = None
    seed: int | None = None
    for evidence in records:
        _check_authoritative(evidence, MIXED_SCHEMA, statuses=("ok", "oom"))
        payload = evidence.payload
        _expect(payload.get("diagnostic_only") is False, f"mixed result is diagnostic: {evidence.path.name}")
        assignment = _mapping(payload.get("assignment"), "mixed.assignment")
        _expect(assignment.get("name") == "A2-P" and assignment.get("starter_commit") == STARTER_COMMIT, "mixed result uses the wrong assignment snapshot")
        model_benchmark = _mapping(payload.get("language_model_benchmark"), "mixed language model benchmark")
        candidate_seed = _integer(payload.get("seed"), "mixed seed")
        model = _string(model_benchmark.get("requested_cuda_model_size"), "mixed model")
        _expect(model in MIXED_MODELS, "mixed model is outside the fixed matrix")
        _expect(model not in indexed, f"duplicate mixed model: {model}")
        _expect(model_benchmark.get("authoritative_cuda_benchmark") is True, "mixed language model benchmark is non-authoritative")
        _expect(model_benchmark.get("same_configuration_and_seed_for_both_precisions") is True, "FP32 and BF16 did not use the same setup")
        _expect(model_benchmark.get("measurement_boundary") == ["forward", "loss", "backward"], "mixed benchmark has the wrong measurement boundary")
        _expect(model_benchmark.get("gradient_clearing_outside_timing") is True, "mixed benchmark timed gradient clearing")
        _expect(model_benchmark.get("optimizer_step_included") is False, "mixed benchmark included an optimizer step")
        config = _mapping(model_benchmark.get("configuration"), "mixed model configuration")
        _expect(config.get("model_size") == model, "mixed model label differs from effective model")
        for key, expected in MODEL_DIMENSIONS[model].items():
            _expect(config.get(key) == expected, f"mixed {model} has wrong {key}")
        _expect(config.get("batch_size") == 4 and config.get("context_length") == 512, "mixed benchmark must use batch 4/context 512")
        _expect(config.get("warmup_steps") >= 5 and config.get("measurement_steps") >= 10, "mixed benchmark has insufficient warmup/samples")
        precision_records = _mapping(model_benchmark.get("records"), "mixed precision records")
        _expect(set(precision_records) == {"fp32", "bf16"}, f"mixed {model} must contain exactly FP32 and BF16 records")
        normalized_records = {precision: _normalize_precision_record(precision_records.get(precision), precision, config) for precision in ("fp32", "bf16")}
        fp32 = normalized_records["fp32"]
        bf16 = normalized_records["bf16"]
        comparison = None
        if fp32["status"] == bf16["status"] == "ok":
            comparison = {
                "bf16_speedup_over_fp32": fp32["timing"]["mean_ms"] / bf16["timing"]["mean_ms"],
                "bf16_forward_speedup_over_fp32": fp32["phase_timings"]["forward_including_loss"]["mean_ms"] / bf16["phase_timings"]["forward_including_loss"]["mean_ms"],
                "bf16_backward_speedup_over_fp32": fp32["phase_timings"]["backward"]["mean_ms"] / bf16["phase_timings"]["backward"]["mean_ms"],
                "first_loss_absolute_difference": abs(fp32["loss"]["first"] - bf16["loss"]["first"]),
                "final_loss_absolute_difference": abs(fp32["loss"]["last"] - bf16["loss"]["last"]),
                "final_logits_mean_absolute_difference": abs(fp32["final_logits"]["mean"] - bf16["final_logits"]["mean"]),
                "peak_allocated_bytes_difference": bf16["memory"]["peak_allocated_bytes"] - fp32["memory"]["peak_allocated_bytes"],
                "peak_reserved_bytes_difference": bf16["memory"]["peak_reserved_bytes"] - fp32["memory"]["peak_reserved_bytes"],
                "peak_active_bytes_difference": bf16["memory"]["peak_active_bytes"] - fp32["memory"]["peak_active_bytes"],
            }
        indexed[model] = {
            "model_size": model,
            "configuration": config,
            "command": _logical_mixed_command(model, candidate_seed),
            "measurement_boundary": ["forward", "loss", "backward"],
            "gradient_clearing_outside_timing": True,
            "optimizer_step_included": False,
            "records": normalized_records,
            "comparison": comparison,
        }
        accumulation = _normalize_accumulation(payload)
        toy = _normalize_toy(payload)
        if shared_accumulation is None:
            shared_accumulation = accumulation
            shared_toy = toy
        else:
            _expect(shared_accumulation == accumulation and shared_toy == toy, "mixed runs disagree on accumulation or ToyModel evidence")
        input_environment = _mapping(payload.get("environment"), "mixed environment")
        candidate_environment = {
            "gpu_model": _string(input_environment.get("gpu_model"), "mixed GPU model"),
            "pytorch_version": _string(input_environment.get("pytorch_version"), "mixed PyTorch version"),
            "pytorch_cuda_version": _string(input_environment.get("pytorch_cuda_version"), "mixed CUDA version"),
        }
        if environment is None:
            environment = candidate_environment
        else:
            _expect(environment == candidate_environment, "mixed runs used inconsistent environments")
        if seed is None:
            seed = candidate_seed
        else:
            _expect(seed == candidate_seed, "mixed runs used inconsistent seeds")

    _expect(set(indexed) == set(MIXED_MODELS), "mixed-precision five-model matrix is incomplete or has extras")
    assert shared_accumulation is not None and shared_toy is not None and environment is not None and seed is not None
    result = {
        "schema_version": PUBLIC_SCHEMA,
        "assignment": {"name": "A2-P", "handout_version": HANDOUT_VERSION, "starter_commit": STARTER_COMMIT},
        "authoritative": True,
        "seed": seed,
        "environment": environment,
        "accumulation": shared_accumulation,
        "toy_model": shared_toy,
        "language_model_benchmarks": [indexed[model] for model in MIXED_MODELS],
    }
    _assert_public(result, "mixed precision public result")
    return result


def _memory_key(payload: dict[str, Any]) -> tuple[str, int, str, str]:
    config = _mapping(payload.get("configuration"), "memory.configuration")
    return (
        _string(config.get("model_size"), "memory model"),
        _integer(config.get("context_length"), "memory context", minimum=1),
        _string(config.get("mode"), "memory mode"),
        _string(config.get("dtype"), "memory dtype"),
    )


def _failure_stage(payload: dict[str, Any]) -> str:
    measurement = _mapping(payload.get("measurement"), "memory measurement")
    if _integer(measurement.get("warmup_completed"), "memory warmup completed", minimum=0) < _integer(
        _mapping(payload.get("configuration"), "memory config").get("warmup"), "memory warmup", minimum=0
    ):
        return "setup_or_warmup"
    boundaries = _sequence(measurement.get("phase_boundaries"), "memory phase boundaries")
    if boundaries:
        last = _mapping(boundaries[-1], "last phase boundary")
        return _string(last.get("label"), "last phase label")
    return "measurement_before_first_phase"


def _validate_stack(value: Any) -> list[dict[str, Any]]:
    stack = []
    for frame in _sequence(value, "maximum allocation stack"):
        item = _mapping(frame, "maximum allocation frame")
        source = _string(item.get("file"), "stack source")
        _expect(not _contains_private_text(source), "maximum allocation stack contains an internal path")
        stack.append(
            {
                "file": source,
                "line": _integer(item.get("line"), "stack line", minimum=0),
                "function": _string(item.get("function"), "stack function"),
            }
        )
    return stack


def _find_case_artifact(evidence_path: Path, basename: str) -> Path | None:
    """Find a basename inside the command's private ``work`` tree.

    ``run_suite.py`` keeps JSON/timelines in ``work/results`` and snapshots in
    ``work/private``.  We intentionally return only a local Path for existence
    checks; it is never serialized into public metadata.
    """

    search_root = evidence_path.parent
    for ancestor in evidence_path.parents:
        if ancestor.name == "work":
            search_root = ancestor
            break
    matches = [path for path in search_root.rglob(basename) if path.is_file() and not path.is_symlink()]
    _expect(len(matches) <= 1, f"ambiguous local artifact basename: {basename}")
    return matches[0] if matches else None


def _validate_timeline(payload: dict[str, Any], label: str) -> tuple[list[dict[str, int | None]], str]:
    snapshot = _mapping(payload.get("snapshot"), f"{label} snapshot")
    summary = _mapping(snapshot.get("summary"), f"{label} snapshot summary")
    points = []
    for point in _sequence(summary.get("timeline_points"), f"{label} timeline points"):
        item = _mapping(point, f"{label} timeline point")
        points.append(
            {
                "event_index": _integer(item.get("event_index"), "timeline event index", minimum=0),
                "time_us": None if item.get("time_us") is None else _integer(item.get("time_us"), "timeline time", minimum=0),
                "active_bytes": _integer(item.get("active_bytes"), "timeline active bytes", minimum=0),
            }
        )
    _expect(len(points) >= 2, f"{label} active-memory timeline is empty")
    axis = _string(summary.get("timeline_x_axis"), f"{label} timeline axis")
    _expect(axis in ("snapshot_time_us", "allocator_event_index"), f"{label} has an unsupported timeline axis")
    if axis == "snapshot_time_us":
        _expect(all(point["time_us"] is not None for point in points), f"{label} timeline timestamps are incomplete")
    return points, axis


def _phase_active_memory_summary(
    measurement: dict[str, Any],
    points: Sequence[dict[str, int | None]],
    axis: str,
    *,
    mode: str,
) -> dict[str, Any]:
    """Project timestamped allocator points into measured phase peak bytes.

    Allocator-event indices cannot be compared with the wall-clock boundaries
    emitted by ``memory_snapshot.py``.  Such runs remain valid, but explicitly
    report that phase attribution is unavailable rather than inventing an
    alignment.
    """

    expected_labels = ["forward"] if mode == "forward" else ["forward", "backward", "optimizer"]
    boundaries: list[dict[str, int | str]] = []
    for value in _sequence(measurement.get("phase_boundaries"), "memory phase boundaries"):
        boundary = _mapping(value, "memory phase boundary")
        start = _integer(boundary.get("start_time_us"), "memory phase start", minimum=0)
        end = _integer(boundary.get("end_time_us"), "memory phase end", minimum=0)
        duration = _integer(boundary.get("duration_us"), "memory phase duration", minimum=0)
        _expect(end >= start and duration == end - start, "memory phase boundary duration is inconsistent")
        boundaries.append(
            {
                "step": _integer(boundary.get("step"), "memory phase step", minimum=0),
                "phase": _string(boundary.get("label"), "memory phase label"),
                "start_time_us": start,
                "end_time_us": end,
            }
        )
    _expect([boundary["phase"] for boundary in boundaries] == expected_labels, f"memory {mode} phase boundaries are incomplete or unordered")

    unavailable = {
        "available": False,
        "timeline_axis": axis,
        "definition": "maximum reconstructed active bytes within each measured phase",
        "peaks": [],
    }
    if axis != "snapshot_time_us":
        return {**unavailable, "unavailable_reason": "allocator events do not expose timestamps for phase alignment"}

    timestamped = [(int(point["time_us"]), int(point["active_bytes"])) for point in points if point["time_us"] is not None]
    _expect(all(left[0] <= right[0] for left, right in zip(timestamped, timestamped[1:], strict=False)), "memory timeline timestamps are unordered")
    peaks: list[dict[str, Any]] = []
    for boundary in boundaries:
        start = int(boundary["start_time_us"])
        end = int(boundary["end_time_us"])
        candidates = [active for timestamp, active in timestamped if start <= timestamp <= end]
        state_at_start = [active for timestamp, active in timestamped if timestamp <= start]
        if state_at_start:
            candidates.append(state_at_start[-1])
        if not candidates:
            return {**unavailable, "unavailable_reason": "timestamped allocator history does not overlap every measured phase"}
        peaks.append(
            {
                "step": int(boundary["step"]),
                "phase": str(boundary["phase"]),
                "active_peak_bytes": max(candidates),
            }
        )
    return {
        "available": True,
        "timeline_axis": axis,
        "definition": "maximum reconstructed active bytes at phase start and after timestamped allocator events within the phase",
        "peaks": peaks,
    }


def _normalize_saved_tensors(payload: dict[str, Any]) -> dict[str, Any] | None:
    value = payload.get("saved_tensors_block")
    if value is None:
        return None
    block = _mapping(value, "saved tensors block")
    _expect(block.get("authoritative") is True, "saved-tensor block analysis is non-authoritative")
    gradients = _mapping(block.get("parameter_gradients"), "saved tensor gradients")
    _expect(gradients.get("matches_theory") is True, "block gradient bytes disagree with theory")
    saved = _mapping(block.get("saved_tensors"), "saved tensors")
    saved_event_count = _integer(saved.get("saved_tensor_event_count"), "saved tensor event count", minimum=1)
    parameter_event_count = _integer(saved.get("parameter_saved_event_count_excluded"), "excluded parameter saved event count", minimum=0)
    _expect(parameter_event_count <= saved_event_count, "excluded parameter saved events exceed all saved events")
    unique_storage_count = _integer(saved.get("unique_saved_storage_count"), "unique saved storage count", minimum=1)
    _expect(unique_storage_count <= saved_event_count - parameter_event_count, "unique saved storage count exceeds non-parameter saved events")
    unique_saved_bytes = _integer(saved.get("unique_saved_bytes"), "unique saved bytes", minimum=1)
    operation_count = _integer(saved.get("operation_count"), "saved tensor operation count", minimum=1)
    release_event_count = _integer(saved.get("release_event_count"), "saved tensor release event count", minimum=0)
    top = []
    for row in _sequence(saved.get("top_5_operations"), "saved tensor top operations"):
        item = _mapping(row, "saved tensor operation")
        source = _string(item.get("source_file"), "saved tensor source")
        _expect(not _contains_private_text(source), "saved tensor source contains an internal path")
        top.append(
            {
                "rank": _integer(item.get("rank"), "saved tensor rank", minimum=1),
                "operation": _string(item.get("operation"), "saved tensor operation"),
                "module_path": _string(item.get("module_path"), "saved tensor module"),
                "source_file": source,
                "source_line": _integer(item.get("source_line"), "saved tensor source line", minimum=0),
                "unique_saved_bytes": _integer(item.get("unique_saved_bytes"), "saved tensor bytes", minimum=0),
                "unique_storage_count": _integer(item.get("unique_storage_count"), "saved tensor storage count", minimum=0),
                "logical_saved_bytes": _integer(item.get("logical_saved_bytes"), "saved tensor logical bytes", minimum=0),
                "saved_event_count": _integer(item.get("saved_event_count"), "saved tensor event count", minimum=0),
                "percent_of_unique_saved_bytes": 100.0 * _integer(item.get("unique_saved_bytes"), "saved tensor bytes", minimum=0) / unique_saved_bytes,
            }
        )
    _expect(len(top) == 5, "saved-tensor analysis must retain the top five operations")
    _expect(operation_count >= len(top), "saved-tensor operation count is smaller than the published top five")
    _expect(sum(int(row["unique_saved_bytes"]) for row in top) <= unique_saved_bytes, "saved-tensor top-five bytes exceed the unique total")
    result = {
        "scope": "one TransformerBlock",
        "configuration": _mapping(block.get("configuration"), "saved tensor configuration"),
        "saved_tensors": {
            "saved_tensor_event_count": saved_event_count,
            "parameter_saved_event_count_excluded": parameter_event_count,
            "unique_saved_storage_count": unique_storage_count,
            "unique_saved_bytes": unique_saved_bytes,
            "operation_count": operation_count,
            "release_event_count": release_event_count,
            "release_event_semantics": "saved-tensor retrievals during backward are release opportunities, not allocator-free observations",
            "top_5_operations": top,
        },
        "parameter_gradients": {
            "formula": "(4*D^2 + 3*D*D_ff + 2*D) * 4 bytes",
            "theoretical_fp32_block_gradient_bytes": _integer(gradients.get("theoretical_fp32_block_gradient_bytes"), "theoretical block gradient bytes", minimum=1),
            "actual_block_gradient_bytes": _integer(gradients.get("actual_block_gradient_bytes"), "actual block gradient bytes", minimum=1),
            "matches_theory": True,
        },
    }
    _assert_public(result, "saved tensors")
    return result


def _logical_memory_command(
    config: dict[str, Any],
    result_basename: str,
    snapshot_basename: str,
    timeline_basename: str,
    memory_viz_basename: str | None,
    saved: bool,
) -> list[str]:
    command = [
        "python",
        "profiling/memory_snapshot.py",
        "--model-size",
        str(config["model_size"]),
        "--batch-size",
        str(config["batch_size"]),
        "--context-length",
        str(config["context_length"]),
        "--mode",
        str(config["mode"]),
        "--dtype",
        str(config["dtype"]),
        "--warmup",
        str(config["warmup"]),
        "--steps",
        str(config["steps"]),
        "--seed",
        str(config["seed"]),
        "--output",
        f"results/memory/{result_basename}",
        "--snapshot-output",
        f"results/memory/{snapshot_basename}",
        "--timeline-output",
        f"results/memory/{timeline_basename}",
    ]
    if memory_viz_basename is not None:
        command.extend(("--memory-viz-output", f"private/{memory_viz_basename}"))
    if saved:
        command.append("--saved-tensors-block")
    return command


def build_memory_outputs(records: Sequence[Evidence]) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, dict[str, Any]]]:
    indexed: dict[tuple[str, int, str, str], Evidence] = {}
    for evidence in records:
        _check_authoritative(evidence, MEMORY_SCHEMA, statuses=("ok", "oom"))
        _expect(evidence.payload.get("dry_run") is False, f"memory result is a dry-run: {evidence.path.name}")
        key = _memory_key(evidence.payload)
        model, context, mode, dtype = key
        _expect(model in ("xl", "large") and mode in MEMORY_MODES and dtype in MEMORY_DTYPES, "memory case is outside the allowed matrix/fallbacks")
        _expect(context in ({128, 1_024, 2_048} if model == "xl" else {2_048}), "memory context is outside the allowed matrix/fallbacks")
        _expect(key not in indexed, f"duplicate memory case: {key}")
        config = _mapping(evidence.payload.get("configuration"), "memory configuration")
        _expect(config.get("batch_size") == 1 and config.get("seed") == 42, "memory must use batch 1 and seed 42")
        _expect(config.get("warmup") >= 5 and config.get("steps") == 1, "memory needs five warmups and one measurement")
        effective = _mapping(evidence.payload.get("effective_configuration"), "effective memory configuration")
        _expect(
            effective.get("model_size") == model and effective.get("context_length") == context and effective.get("device") == "cuda" and effective.get("is_reduced") is False,
            "memory configuration was silently reduced",
        )
        indexed[key] = evidence

    primary = {("xl", context, mode, dtype) for context in MEMORY_CONTEXTS for mode in MEMORY_MODES for dtype in MEMORY_DTYPES}
    _expect(primary.issubset(indexed), "memory primary XL matrix is incomplete")
    for mode in MEMORY_MODES:
        for dtype in MEMORY_DTYPES:
            _expect(indexed[("xl", 128, mode, dtype)].payload.get("status") == "ok", f"XL/context 128 {mode}/{dtype} must succeed")
    allowed = set(primary)
    fallback_for: dict[tuple[str, int, str, str], tuple[str, int, str, str]] = {}
    for mode in MEMORY_MODES:
        for dtype in MEMORY_DTYPES:
            original = ("xl", 2_048, mode, dtype)
            if indexed[original].payload.get("status") != "oom":
                continue
            first = ("xl", 1_024, mode, dtype)
            second = ("large", 2_048, mode, dtype)
            _expect(first in indexed, f"OOM {original} is missing XL/context 1024 fallback")
            allowed.add(first)
            fallback_for[first] = original
            if indexed[first].payload.get("status") == "oom":
                _expect(second in indexed and indexed[second].payload.get("status") == "ok", f"OOM {first} is missing successful Large/context 2048 fallback")
                allowed.add(second)
                fallback_for[second] = original
            else:
                _expect(indexed[first].payload.get("status") == "ok", f"fallback {first} did not succeed or OOM")
    _expect(set(indexed) == allowed, "memory evidence contains unrequested or unordered fallback cases")

    rows: list[dict[str, Any]] = []
    metadata_runs: list[dict[str, Any]] = []
    timelines: dict[str, dict[str, Any]] = {}
    environment: dict[str, Any] | None = None
    saved_analysis: dict[str, Any] | None = None
    order = sorted(indexed, key=lambda key: (0 if key in primary else 1, key[1], MEMORY_MODES.index(key[2]), MEMORY_DTYPES.index(key[3]), key[0]))
    for key in order:
        evidence = indexed[key]
        payload = evidence.payload
        config = _mapping(payload.get("configuration"), "memory configuration")
        model, context, mode, dtype = key
        status = _string(payload.get("status"), "memory status")
        measurement = _mapping(payload.get("measurement"), "memory measurement")
        warmup_completed = _integer(measurement.get("warmup_completed"), "memory warmup completed", minimum=0)
        steps_completed = _integer(measurement.get("steps_completed"), "memory steps completed", minimum=0)
        memory = _mapping(payload.get("memory"), "memory counters")
        metrics: dict[str, int | None] = {}
        for field in (
            "active_bytes_current",
            "active_bytes_peak",
            "allocated_bytes_current",
            "allocated_bytes_peak",
            "reserved_bytes_current",
            "reserved_bytes_peak",
            "max_memory_allocated",
            "maximum_single_allocation_bytes",
        ):
            metrics[field] = _optional_nonnegative_integer(memory.get(field), f"memory {field}")
        if status == "ok":
            _expect(warmup_completed >= 5 and steps_completed == 1, f"successful memory case {key} has incomplete execution")
            _expect(measurement.get("history_started_after_warmup") is True, f"memory history boundary is wrong for {key}")
            _expect(all(value is not None for value in metrics.values()), f"successful memory case {key} lacks peak counters")
            _expect(
                all(
                    int(metrics[field] or 0) > 0
                    for field in ("active_bytes_peak", "allocated_bytes_peak", "reserved_bytes_peak", "max_memory_allocated", "maximum_single_allocation_bytes")
                ),
                f"successful memory case {key} has empty peak evidence",
            )
            artifacts = _mapping(payload.get("artifacts"), "memory artifacts")
            _expect(artifacts.get("snapshot_generated") is True and artifacts.get("timeline_generated") is True, f"successful memory case {key} lacks snapshot/timeline")
            _expect(artifacts.get("timeline_derived_from_same_snapshot") is True, f"successful memory case {key} timeline is not tied to its snapshot")
            points, axis = _validate_timeline(payload, f"memory {key}")
        else:
            exception = _mapping(payload.get("exception"), "memory OOM exception")
            _string(exception.get("type"), "memory OOM exception type")
            points, axis = [], "allocator_event_index"
        stack = _validate_stack(memory.get("maximum_single_allocation_stack", []))
        if status == "ok":
            _expect(stack, f"successful memory case {key} lacks the maximum-allocation stack")
        d_model = MODEL_DIMENSIONS[model]["d_model"]
        elements = int(config["batch_size"]) * context * d_model
        original = fallback_for.get(key)
        fallback_label = "" if original is None else f"{original[0]}_ctx{original[1]}_{original[2]}_{original[3]}"
        case_id = f"{model}_b1_ctx{context}_{mode}_{dtype}"
        exception_type = ""
        failure_stage = ""
        if status == "oom":
            exception_type = _string(_mapping(payload.get("exception"), "memory exception").get("type"), "memory exception type")
            failure_stage = _failure_stage(payload)
        row = {
            "case_id": case_id,
            "model_size": model,
            "batch_size": 1,
            "context_length": context,
            "mode": mode,
            "dtype": dtype,
            "status": status,
            "is_fallback": original is not None,
            "fallback_for": fallback_label,
            "warmup_completed": warmup_completed,
            "measurement_steps_completed": steps_completed,
            "measured_elapsed_ms": _finite_float_text(measurement.get("measured_elapsed_ms"), "memory elapsed", optional=True),
            **metrics,
            "residual_stream_elements": elements,
            "residual_stream_theory_formula": "B*T*D*bytes_per_element",
            "residual_stream_fp32_bytes": elements * 4,
            "residual_stream_bf16_bytes": elements * 2,
            "exception_type": exception_type,
            "failure_stage": failure_stage,
        }
        _assert_public(row, "memory row")
        rows.append(row)
        artifacts = _mapping(payload.get("artifacts"), "memory artifacts")
        result_basename = _safe_basename(evidence.path.name, "memory result filename")
        snapshot_basename = _safe_basename(artifacts.get("snapshot_basename"), "memory snapshot filename")
        timeline_basename = _safe_basename(artifacts.get("timeline_basename"), "memory timeline filename")
        snapshot_generated = artifacts.get("snapshot_generated") is True
        timeline_generated = artifacts.get("timeline_generated") is True
        memory_viz_basename_value = artifacts.get("memory_viz_basename")
        memory_viz_basename = None if memory_viz_basename_value is None else _safe_basename(memory_viz_basename_value, "official memory visualizer filename")
        memory_viz_generated = artifacts.get("memory_viz_generated") is True
        if snapshot_generated:
            _expect(_find_case_artifact(evidence.path, snapshot_basename) is not None, f"recorded snapshot is missing: {snapshot_basename}")
        if timeline_generated:
            _expect(_find_case_artifact(evidence.path, timeline_basename) is not None, f"recorded timeline is missing: {timeline_basename}")
        if memory_viz_generated:
            _expect(memory_viz_basename is not None, f"official memory visualizer basename is missing for {key}")
            _expect(_find_case_artifact(evidence.path, memory_viz_basename) is not None, f"recorded official memory visualizer is missing: {memory_viz_basename}")
        elif memory_viz_basename is not None and status == "ok":
            raise SummaryError(f"requested official memory visualizer was not generated for {key}")
        normalized_saved = _normalize_saved_tensors(payload)
        if normalized_saved is not None:
            _expect(saved_analysis is None, "saved-tensor block analysis appears in multiple runs")
            saved_analysis = {"source_case_id": case_id, **normalized_saved}
        metadata_runs.append(
            {
                "case_id": case_id,
                "configuration": {
                    "model_size": model,
                    "batch_size": 1,
                    "context_length": context,
                    "mode": mode,
                    "dtype": dtype,
                    "parameter_dtype": "fp32",
                    "warmup_steps": int(config["warmup"]),
                    "measurement_steps": int(config["steps"]),
                    "seed": int(config["seed"]),
                },
                "status": status,
                "is_fallback": original is not None,
                "fallback_for": fallback_label or None,
                "command": _logical_memory_command(
                    config,
                    result_basename,
                    snapshot_basename,
                    timeline_basename,
                    memory_viz_basename,
                    normalized_saved is not None,
                ),
                "local_snapshot_basename": snapshot_basename,
                "local_snapshot_generated": snapshot_generated,
                "snapshot_published": False,
                "local_timeline_basename": timeline_basename,
                "local_official_memory_viz_basename": memory_viz_basename,
                "local_official_memory_viz_generated": memory_viz_generated,
                "official_memory_viz_published": False,
                "public_timeline_asset": None,
                "maximum_single_allocation_bytes": metrics["maximum_single_allocation_bytes"],
                "maximum_single_allocation_stack": stack,
                "exception_type": exception_type or None,
                "failure_stage": failure_stage or None,
            }
        )
        runtime = _mapping(payload.get("runtime"), "memory runtime")
        candidate_environment = {
            "gpu_model": _string(runtime.get("gpu_model"), "memory GPU model"),
            "pytorch_version": _string(runtime.get("torch_version"), "memory PyTorch version"),
            "cuda_runtime_version": _string(runtime.get("cuda_runtime_version"), "memory CUDA version"),
        }
        if environment is None:
            environment = candidate_environment
        else:
            _expect(environment == candidate_environment, "memory runs used inconsistent environments")
        if status == "ok":
            timelines[case_id] = {
                "points": points,
                "axis": axis,
                "configuration": metadata_runs[-1]["configuration"],
                "phase_boundaries": measurement.get("phase_boundaries", []),
                "official_memory_viz_generated": memory_viz_generated,
            }
            metadata_runs[-1]["phase_active_memory"] = _phase_active_memory_summary(measurement, points, axis, mode=mode)
        else:
            metadata_runs[-1]["phase_active_memory"] = {
                "available": False,
                "timeline_axis": None,
                "definition": "maximum reconstructed active bytes within each measured phase",
                "peaks": [],
                "unavailable_reason": "memory case did not complete successfully",
            }

    _expect(saved_analysis is not None, "memory evidence lacks authoritative one-block saved-tensor analysis")
    selected_timeline_ids: dict[str, str] = {}
    for mode in MEMORY_MODES:
        candidates = [
            f"xl_b1_ctx2048_{mode}_fp32",
            f"xl_b1_ctx1024_{mode}_fp32",
            f"large_b1_ctx2048_{mode}_fp32",
        ]
        selected = next((case_id for case_id in candidates if case_id in timelines), None)
        _expect(selected is not None, f"no successful FP32 timeline is available for {mode}")
        _expect(timelines[selected]["official_memory_viz_generated"] is True, f"selected {mode} timeline lacks the official private memory visualizer evidence")
        selected_timeline_ids[mode] = selected
    public_assets = {
        "forward": "assets/memory_forward_active_timeline.png",
        "train_step": "assets/memory_train_step_active_timeline.png",
    }
    for run in metadata_runs:
        mode = str(run["configuration"]["mode"])
        if run["case_id"] == selected_timeline_ids.get(mode):
            run["public_timeline_asset"] = public_assets[mode]
    assert environment is not None
    metadata = {
        "schema_version": PUBLIC_SCHEMA,
        "assignment": "A2-P",
        "handout_version": HANDOUT_VERSION,
        "starter_commit": STARTER_COMMIT,
        "environment": environment,
        "memory_counter_definitions": {
            "active": "torch.cuda.memory_stats active_bytes.all current/peak",
            "allocated": "torch.cuda.memory_stats allocated_bytes.all current/peak",
            "reserved": "torch.cuda.memory_stats reserved_bytes.all current/peak",
            "max_memory_allocated": "torch.cuda.max_memory_allocated",
        },
        "history_boundary": "allocator history starts after all warmup steps",
        "residual_stream_theory": {
            "formula": "B*T*D*4 bytes for one FP32 residual-stream tensor",
            "bf16_variant": "B*T*D*2 bytes for one BF16 tensor",
            "pinned_pdf_reference": {
                "batch_size": 4,
                "context_length": 512,
                "d_model": 2_560,
                "fp32_bytes": 20 * 1024 * 1024,
                "fp32_mib": 20.0,
            },
        },
        "raw_artifact_policy": {"snapshots_retained_locally": True, "snapshots_published": False},
        "fallback_order": ["XL/context 1024", "Large/context 2048"],
        "runs": metadata_runs,
        "selected_timeline_cases": selected_timeline_ids,
        "saved_tensors_block": saved_analysis,
    }
    _assert_public(metadata, "memory metadata")
    return rows, metadata, {mode: timelines[case_id] for mode, case_id in selected_timeline_ids.items()}


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        with temporary.open("x", encoding="utf-8", newline="") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _write_json(path: Path, value: Any) -> None:
    _assert_public(value)
    _atomic_write_text(path, json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")


def _csv_text(fields: Sequence[str], rows: Sequence[Mapping[str, Any]]) -> str:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=list(fields), extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    for row in rows:
        _expect(set(row) == set(fields), "CSV row does not match its public schema")
        _assert_public(dict(row), "CSV row")
        writer.writerow({field: "" if row[field] is None else row[field] for field in fields})
    return buffer.getvalue()


def _prepare_matplotlib() -> Any:
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "cs336-a2p-matplotlib"))
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    return plt


def _save_optimized_png(figure: Any, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    raw = output.with_name(f".{output.stem}.raw.png")
    temporary = output.with_name(f".{output.stem}.tmp.png")
    try:
        figure.savefig(raw, dpi=130, metadata={"Software": "CS336 A2-P summarizer"}, facecolor="white")
        from PIL import Image

        with Image.open(raw) as image:
            image = image.convert("RGB")
            image.thumbnail((1_600, 1_000))
            image.save(temporary, format="PNG", optimize=True, compress_level=9)
        os.replace(temporary, output)
    finally:
        raw.unlink(missing_ok=True)
        temporary.unlink(missing_ok=True)


def _iter_chrome_trace_events(path: Path) -> Iterable[dict[str, Any]]:
    """Stream objects from a Chrome trace's top-level ``traceEvents`` array."""

    _expect(path.is_file() and not path.is_symlink(), "representative Chrome trace is missing or symlinked")
    _expect(path.stat().st_size > 0, "representative Chrome trace is empty")
    decoder = json.JSONDecoder()
    buffer = ""
    cursor = 0
    eof = False
    array_found = False
    key_pattern = re.compile(r'"traceEvents"\s*:\s*\[')
    try:
        with path.open("r", encoding="utf-8") as handle:
            while not array_found:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                buffer += chunk
                match = key_pattern.search(buffer)
                if match is not None:
                    cursor = match.end()
                    array_found = True
                    break
                if len(buffer) > 2 * 1024 * 1024:
                    buffer = buffer[-256:]
            _expect(array_found, "Chrome trace has no traceEvents array")

            while True:
                while True:
                    while cursor < len(buffer) and buffer[cursor] in " \t\r\n,":
                        cursor += 1
                    if cursor < len(buffer) or eof:
                        break
                    chunk = handle.read(1024 * 1024)
                    if chunk:
                        buffer += chunk
                    else:
                        eof = True
                _expect(cursor < len(buffer), "Chrome trace ended before traceEvents closed")
                if buffer[cursor] == "]":
                    return
                while True:
                    try:
                        value, end = decoder.raw_decode(buffer, cursor)
                        break
                    except json.JSONDecodeError as error:
                        chunk = handle.read(1024 * 1024)
                        if not chunk:
                            raise SummaryError("Chrome trace has a malformed traceEvents entry") from error
                        buffer += chunk
                cursor = end
                if isinstance(value, dict):
                    yield value
                if cursor > 4 * 1024 * 1024:
                    buffer = buffer[cursor:]
                    cursor = 0
    except (OSError, UnicodeError) as error:
        raise SummaryError(f"cannot stream representative Chrome trace: {type(error).__name__}") from error


def _normalized_trace_event(value: dict[str, Any]) -> dict[str, Any] | None:
    if value.get("ph") != "X":
        return None
    name = value.get("name")
    category = value.get("cat", "")
    if not isinstance(name, str) or not name or not isinstance(category, str):
        return None
    try:
        timestamp = float(value.get("ts"))
        duration = float(value.get("dur"))
    except (TypeError, ValueError):
        return None
    if not math.isfinite(timestamp) or not math.isfinite(duration) or duration < 0.0:
        return None
    pid = value.get("pid")
    tid = value.get("tid")
    if type(pid) not in (int, str) or type(tid) not in (int, str):
        return None
    return {"name": name, "category": category, "ts_us": timestamp, "duration_us": duration, "pid": str(pid), "tid": str(tid)}


def _trace_event_kind(event: dict[str, Any]) -> str | None:
    category = str(event["category"]).lower()
    if "kernel" in category or category in {"gpu", "cuda_kernel"}:
        return "cuda_kernel"
    if "cpu_op" in category or category in {"operator", "pytorch function"}:
        return "cpu_operator"
    return None


def _is_cpu_user_annotation(event: dict[str, Any]) -> bool:
    return str(event["category"]).strip().lower() == "user_annotation"


def extract_profile_kernel_aggregates(path: Path) -> list[dict[str, Any]]:
    """Aggregate real ``cat=kernel`` events from one measured profiler step.

    Chrome traces can be very large, so both passes stream the top-level event
    array.  The first pass finds the measured-step time window; the second
    aggregates kernels by their exact, privacy-checked name.  Only the bounded
    top-N projection enters the public CSV, while each reported CUDA duration
    remains the sum of source ``dur`` values in microseconds.
    """

    measure_events: list[dict[str, Any]] = []
    for raw in _iter_chrome_trace_events(path):
        event = _normalized_trace_event(raw)
        if event is not None and event["name"] == "profile/measure" and _is_cpu_user_annotation(event):
            measure_events.append(event)
    _expect(measure_events, "Chrome trace is missing the profile/measure range")
    measure = max(measure_events, key=lambda event: float(event["duration_us"]))
    measure_start = float(measure["ts_us"])
    measure_end = measure_start + float(measure["duration_us"])
    _expect(measure_end > measure_start, "Chrome trace profile/measure range has no positive duration")

    aggregates: dict[str, dict[str, float | int]] = {}
    for raw in _iter_chrome_trace_events(path):
        event = _normalized_trace_event(raw)
        if event is None or str(event["category"]).strip().lower() != "kernel":
            continue
        start = float(event["ts_us"])
        duration = float(event["duration_us"])
        end = start + duration
        if duration <= 0.0 or end <= measure_start or start >= measure_end:
            continue
        name = str(event["name"])
        _expect(len(name) <= MAX_PROFILE_KERNEL_NAME_LENGTH, "CUDA kernel name is unexpectedly long")
        _expect(not _contains_private_text(name), "private metadata found in a CUDA kernel name")
        if name not in aggregates:
            _expect(
                len(aggregates) < MAX_PROFILE_UNIQUE_KERNEL_NAMES,
                "Chrome trace has too many unique CUDA kernel names",
            )
            aggregates[name] = {"calls": 0, "cuda_time_us": 0.0}
        aggregates[name]["calls"] = int(aggregates[name]["calls"]) + 1
        aggregates[name]["cuda_time_us"] = float(aggregates[name]["cuda_time_us"]) + duration

    _expect(aggregates, "Chrome trace has no positive cat=kernel events inside profile/measure")
    ranked = sorted(
        aggregates.items(),
        key=lambda item: (-float(item[1]["cuda_time_us"]), -int(item[1]["calls"]), item[0]),
    )[:MAX_PROFILE_KERNEL_ROWS_PER_RUN]
    return [
        {
            "scope": "profiler_trace_kernel",
            "op_or_kernel": "cuda_kernel",
            "name": name,
            "calls": int(values["calls"]),
            "cpu_time_us": None,
            "cuda_time_us": float(values["cuda_time_us"]),
        }
        for name, values in ranked
    ]


def extract_trace_timeline(path: Path) -> dict[str, Any]:
    """Extract a bounded, trace-derived reconstruction of one measured step."""

    phase_events: dict[str, list[dict[str, Any]]] = {name: [] for name in REQUIRED_RANGES}
    for raw in _iter_chrome_trace_events(path):
        event = _normalized_trace_event(raw)
        if event is not None and event["name"] in phase_events and _is_cpu_user_annotation(event):
            phase_events[event["name"]].append(event)
    missing = [name for name, events in phase_events.items() if not events]
    _expect(not missing, f"representative Chrome trace is missing record_function ranges: {missing}")
    measure = max(phase_events["profile/measure"], key=lambda event: float(event["duration_us"]))
    measure_start = float(measure["ts_us"])
    measure_end = measure_start + float(measure["duration_us"])
    _expect(measure_end > measure_start, "profile/measure has no positive duration in the Chrome trace")

    cpu_candidates: list[dict[str, Any]] = []
    kernel_candidates: list[dict[str, Any]] = []
    for raw in _iter_chrome_trace_events(path):
        event = _normalized_trace_event(raw)
        if event is None or _contains_private_text(str(event["name"])):
            continue
        start = float(event["ts_us"])
        end = start + float(event["duration_us"])
        if end < measure_start or start > measure_end:
            continue
        kind = _trace_event_kind(event)
        if kind == "cuda_kernel":
            kernel_candidates.append(event)
        elif kind == "cpu_operator":
            cpu_candidates.append(event)
    _expect(cpu_candidates, "representative Chrome trace has no CPU operator events inside profile/measure")
    _expect(kernel_candidates, "representative Chrome trace has no CUDA kernel events inside profile/measure")

    cpu_events = sorted(cpu_candidates, key=lambda event: (-float(event["duration_us"]), float(event["ts_us"]), str(event["name"])))[:80]
    kernel_events = sorted(kernel_candidates, key=lambda event: (-float(event["duration_us"]), float(event["ts_us"]), str(event["name"])))[:400]
    stream_totals: dict[tuple[str, str], float] = {}
    for event in kernel_events:
        key = (str(event["pid"]), str(event["tid"]))
        stream_totals[key] = stream_totals.get(key, 0.0) + float(event["duration_us"])
    selected_streams = [key for key, _total in sorted(stream_totals.items(), key=lambda item: (-item[1], item[0]))[:4]]
    kernel_events = [event for event in kernel_events if (str(event["pid"]), str(event["tid"])) in selected_streams]
    _expect(kernel_events, "representative Chrome trace has no CUDA kernels after bounded track selection")

    all_phase_starts = [float(event["ts_us"]) for events in phase_events.values() for event in events]
    all_phase_ends = [float(event["ts_us"]) + float(event["duration_us"]) for events in phase_events.values() for event in events]
    plot_start = min(measure_start, min(all_phase_starts))
    plot_end = max(measure_end, max(all_phase_ends))
    _expect(plot_end > plot_start, "representative trace crop is empty")
    return {
        "phase_events": phase_events,
        "cpu_events": cpu_events,
        "kernel_events": kernel_events,
        "selected_streams": selected_streams,
        "measure_start_us": measure_start,
        "measure_end_us": measure_end,
        "plot_start_us": plot_start,
        "plot_end_us": plot_end,
        "source_event_counts": {"cpu_operator": len(cpu_candidates), "cuda_kernel": len(kernel_candidates)},
    }


def _timeline_span(event: dict[str, Any], origin_us: float, end_us: float) -> tuple[float, float] | None:
    start = max(origin_us, float(event["ts_us"]))
    end = min(end_us, float(event["ts_us"]) + float(event["duration_us"]))
    if end <= start:
        return None
    return (start - origin_us) / 1_000.0, (end - start) / 1_000.0


def render_compute_profile(representative: dict[str, Any], output: Path) -> dict[str, Any]:
    trace_path = representative.get("trace_path")
    _expect(isinstance(trace_path, Path), "representative trace path was not retained locally")
    timeline = extract_trace_timeline(trace_path)
    origin = float(timeline["plot_start_us"])
    plot_end = float(timeline["plot_end_us"])
    plt = _prepare_matplotlib()
    figure, axis = plt.subplots(figsize=(13.2, 8.0))
    phase_colors = {
        "profile/warmup": "#7f7f7f",
        "profile/measure": "#1f77b4",
        "forward": "#2ca02c",
        "backward": "#d62728",
        "optimizer": "#9467bd",
        "attention/scores": "#bcbd22",
        "attention/softmax": "#17becf",
        "attention/value": "#ff7f0e",
    }
    labels: list[str] = []
    y = 0
    for name in REQUIRED_RANGES:
        spans = [span for event in timeline["phase_events"][name] if (span := _timeline_span(event, origin, plot_end)) is not None]
        axis.broken_barh(spans, (y - 0.36, 0.72), facecolors=phase_colors[name], alpha=0.9)
        labels.append(name)
        y += 1

    cpu_spans = [span for event in timeline["cpu_events"] if (span := _timeline_span(event, origin, plot_end)) is not None]
    axis.broken_barh(cpu_spans, (y - 0.36, 0.72), facecolors="#4c78a8", alpha=0.75)
    labels.append("CPU ops (80 longest)")
    y += 1
    for stream_index, stream in enumerate(timeline["selected_streams"], start=1):
        events = [event for event in timeline["kernel_events"] if (str(event["pid"]), str(event["tid"])) == stream]
        spans = [span for event in events if (span := _timeline_span(event, origin, plot_end)) is not None]
        axis.broken_barh(spans, (y - 0.36, 0.72), facecolors="#e45756", alpha=0.82)
        labels.append(f"CUDA kernel track {stream_index}")
        y += 1

    measure_start_ms = (float(timeline["measure_start_us"]) - origin) / 1_000.0
    measure_end_ms = (float(timeline["measure_end_us"]) - origin) / 1_000.0
    axis.axvline(measure_start_ms, color="#1f77b4", linestyle="--", linewidth=0.9)
    axis.axvline(measure_end_ms, color="#1f77b4", linestyle="--", linewidth=0.9)
    axis.set_yticks(range(len(labels)), labels=labels)
    axis.invert_yaxis()
    axis.set_xlabel("elapsed trace timestamp (ms; cropped around profile/measure)")
    axis.set_title(f"Trace-derived compute timeline — {representative['run_id']}")
    axis.grid(True, axis="x", alpha=0.2)
    longest = sorted(timeline["kernel_events"], key=lambda event: (-float(event["duration_us"]), str(event["name"])))[:5]
    longest_text = "Longest CUDA events in crop:\n" + "\n".join(
        f"{index}. {str(event['name'])[:72]} — {float(event['duration_us']) / 1_000.0:.3f} ms" for index, event in enumerate(longest, start=1)
    )
    figure.text(0.995, 0.01, longest_text, ha="right", va="bottom", fontsize=7, family="monospace")
    figure.suptitle("Reconstructed from the local torch.profiler Chrome trace; raw trace is not published", fontsize=9, y=0.995)
    figure.tight_layout(rect=(0.0, 0.11, 1.0, 0.98))
    try:
        _save_optimized_png(figure, output)
    finally:
        plt.close(figure)
    summary = {
        "kind": "trace-derived reconstruction",
        "source": "local torch.profiler Chrome trace retained privately",
        "raw_trace_published": False,
        "measure_duration_ms": (float(timeline["measure_end_us"]) - float(timeline["measure_start_us"])) / 1_000.0,
        "required_range_event_counts": {name: len(timeline["phase_events"][name]) for name in REQUIRED_RANGES},
        "cpu_operator_events_in_measure": int(timeline["source_event_counts"]["cpu_operator"]),
        "cuda_kernel_events_in_measure": int(timeline["source_event_counts"]["cuda_kernel"]),
        "rendered_cuda_tracks": len(timeline["selected_streams"]),
    }
    _assert_public(summary, "trace reconstruction summary")
    return summary


def _downsample_timeline(points: list[dict[str, Any]], maximum_points: int = 20_000) -> list[dict[str, Any]]:
    """Bound plotting cost while retaining bin endpoints and active-byte extrema."""

    if len(points) <= maximum_points:
        return points
    bins = max(1, maximum_points // 4)
    width = math.ceil(len(points) / bins)
    retained: set[int] = {0, len(points) - 1}
    for start in range(0, len(points), width):
        stop = min(len(points), start + width)
        indexes = range(start, stop)
        retained.update(
            {
                start,
                stop - 1,
                min(indexes, key=lambda index: int(points[index]["active_bytes"])),
                max(indexes, key=lambda index: int(points[index]["active_bytes"])),
            }
        )
    return [points[index] for index in sorted(retained)]


def render_memory_timeline(timeline: dict[str, Any], output: Path) -> None:
    points = _downsample_timeline(timeline["points"])
    axis_kind = timeline["axis"]
    config = timeline["configuration"]
    if axis_kind == "snapshot_time_us":
        origin = min(int(point["time_us"]) for point in points)
        x_values = [(int(point["time_us"]) - origin) / 1_000.0 for point in points]
        x_label = "elapsed allocator-history time (ms)"
    else:
        origin = 0
        x_values = [int(point["event_index"]) for point in points]
        x_label = "allocator event index"
    y_values = [int(point["active_bytes"]) / 2**20 for point in points]
    plt = _prepare_matplotlib()
    figure, axis = plt.subplots(figsize=(10.5, 4.8))
    axis.step(x_values, y_values, where="post", linewidth=1.35, color="#3465a4", label="active allocator bytes")
    axis.fill_between(x_values, y_values, step="post", color="#3465a4", alpha=0.16)
    colors = {"forward": "#2ca02c", "backward": "#d62728", "optimizer": "#9467bd"}
    labels_seen: set[str] = set()
    if axis_kind == "snapshot_time_us":
        low, high = min(int(point["time_us"]) for point in points), max(int(point["time_us"]) for point in points)
        tolerance = max(1_000, high - low)
        for boundary in _sequence(timeline.get("phase_boundaries", []), "timeline phase boundaries"):
            item = _mapping(boundary, "timeline phase boundary")
            label = _string(item.get("label"), "timeline phase label")
            start = _integer(item.get("start_time_us"), "timeline phase start", minimum=0)
            if label not in colors or not (low - tolerance <= start <= high + tolerance):
                continue
            legend = label if label not in labels_seen else None
            labels_seen.add(label)
            axis.axvline((start - origin) / 1_000.0, color=colors[label], linestyle="--", linewidth=0.9, label=legend)
    axis.set_xlabel(x_label)
    axis.set_ylabel("active memory (MiB)")
    axis.set_title(f"Active Memory Timeline — {config['model_size']} / context {config['context_length']} / {config['mode']} / {config['dtype']}")
    axis.grid(True, alpha=0.2)
    axis.legend(loc="best")
    figure.tight_layout()
    try:
        _save_optimized_png(figure, output)
    finally:
        plt.close(figure)


def _attachment_size(root: Path) -> int:
    files = [path for folder in (root / "results", root / "assets") if folder.exists() for path in folder.rglob("*") if path.is_file()]
    return sum(path.stat().st_size for path in files)


def _validate_staged_tree(root: Path) -> int:
    actual = {path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()}
    _expect(
        actual == set(PUBLIC_FILES), f"public staging tree differs from required files: missing={sorted(set(PUBLIC_FILES) - actual)}, extra={sorted(actual - set(PUBLIC_FILES))}"
    )
    for path in actual:
        _expect(not path.endswith((".trace.json", ".pickle", ".nsys-rep", ".sqlite")), "raw profiler artifact entered the public tree")
    assets = {path for path in actual if path.startswith("assets/")}
    _expect(assets == PUBLIC_ASSETS and len(assets) == 3, "public bundle must contain exactly three managed images")
    size = _attachment_size(root)
    _expect(size <= ATTACHMENT_BUDGET_BYTES, f"results/assets use {size} bytes, above the 2 MiB limit")
    return size


def _publish(staged: Path, public_root: Path) -> None:
    public_root.mkdir(parents=True, exist_ok=True)
    _expect(not public_root.is_symlink(), "--public-root must not be a symlink")
    existing_assets = {path.relative_to(public_root).as_posix() for path in (public_root / "assets").rglob("*") if path.is_file()} if (public_root / "assets").exists() else set()
    existing_results = (
        {path.relative_to(public_root).as_posix() for path in (public_root / "results").rglob("*") if path.is_file()} if (public_root / "results").exists() else set()
    )
    managed_results = {path for path in PUBLIC_FILES if path.startswith("results/")}
    _expect(existing_assets.issubset(PUBLIC_ASSETS), f"refusing to remove unmanaged public assets: {sorted(existing_assets - PUBLIC_ASSETS)}")
    _expect(existing_results.issubset(managed_results), f"refusing to remove unmanaged public results: {sorted(existing_results - managed_results)}")
    for relative in PUBLIC_FILES:
        source = staged / relative
        destination = public_root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(f".{destination.name}.publish-{os.getpid()}")
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
    for stale in existing_assets - PUBLIC_ASSETS:
        (public_root / stale).unlink(missing_ok=True)


def build_public_bundle(raw_root: Path, public_root: Path, *, dry_run: bool = False) -> BuildSummary:
    validate_suite_manifest(raw_root)
    evidence = discover_evidence(raw_root)
    benchmark_records = [record for record in evidence[BENCHMARK_SCHEMA] if record.payload.get("profile") is None]
    profile_records = [record for record in evidence[BENCHMARK_SCHEMA] if isinstance(record.payload.get("profile"), dict)]
    benchmark_rows, benchmark_environment = build_benchmark_rows(benchmark_records)
    profile_rows, profile_metadata, representative = build_profile_outputs(profile_records)
    _expect(profile_metadata["environment"] == benchmark_environment, "benchmark and profile environments differ")
    mixed_payload = build_mixed_payload(evidence[MIXED_SCHEMA])
    _expect(mixed_payload["environment"]["gpu_model"] == benchmark_environment["gpu_model"], "mixed and benchmark GPU models differ")
    memory_rows, memory_metadata, timelines = build_memory_outputs(evidence[MEMORY_SCHEMA])
    _expect(memory_metadata["environment"]["gpu_model"] == benchmark_environment["gpu_model"], "memory and benchmark GPU models differ")

    target_parent = public_root.expanduser().resolve().parent
    target_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".a2p-summary-", dir=target_parent) as temporary:
        staged = Path(temporary)
        _atomic_write_text(staged / "results/benchmark.csv", _csv_text(BENCHMARK_FIELDS, benchmark_rows))
        _atomic_write_text(staged / "results/profile/trace_summary.csv", _csv_text(PROFILE_FIELDS, profile_rows))
        profile_metadata["representative_trace_reconstruction"] = render_compute_profile(representative, staged / "assets/compute_profile.png")
        _write_json(staged / "results/profile/run_metadata.json", profile_metadata)
        _write_json(staged / "results/mixed_precision.json", mixed_payload)
        _atomic_write_text(staged / "results/memory/peaks.csv", _csv_text(MEMORY_FIELDS, memory_rows))
        _write_json(staged / "results/memory/run_metadata.json", memory_metadata)
        render_memory_timeline(timelines["forward"], staged / "assets/memory_forward_active_timeline.png")
        render_memory_timeline(timelines["train_step"], staged / "assets/memory_train_step_active_timeline.png")
        attachment_bytes = _validate_staged_tree(staged)
        if not dry_run:
            _publish(staged, public_root.expanduser().resolve())
    return BuildSummary(
        benchmark_samples=len(benchmark_rows),
        profile_runs=6,
        profile_rows=len(profile_rows),
        mixed_models=5,
        memory_cases=len(memory_rows),
        attachment_bytes=attachment_bytes,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Validate A2-P raw evidence and build the public results/assets bundle.")
    parser.add_argument("--raw-root", type=Path, required=True, help="private suite output containing JSON, traces, snapshots and timelines")
    parser.add_argument("--public-root", type=Path, required=True, help="student A2-P directory that will receive results/ and assets/")
    parser.add_argument("--dry-run", action="store_true", help="validate, render and size-check in a temporary directory without publishing")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        summary = build_public_bundle(args.raw_root, args.public_root, dry_run=args.dry_run)
    except SummaryError as error:
        print(json.dumps({"status": "error", "error_type": type(error).__name__, "error": str(error)}, ensure_ascii=False, sort_keys=True))
        return 2
    output = summary.as_dict()
    output["dry_run"] = args.dry_run
    print(json.dumps(output, ensure_ascii=False, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
