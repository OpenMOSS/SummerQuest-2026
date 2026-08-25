#!/usr/bin/env python3
"""Validate private A2-K raw artifacts and create the public result bundle.

No public file is written until all required raw artifacts, matrix rows,
runtime guards, and hardware classifications have passed validation.  A run on
H200 or any other development GPU can be summarized, but every resulting file
is explicitly classified as non-authoritative and never presented as RTX 4090
24GB evidence.
"""

from __future__ import annotations

import argparse
from collections.abc import Iterable, Mapping, Sequence
import copy
import csv
from io import StringIO
import json
import math
import os
from pathlib import Path
import re
import statistics
import tempfile
from typing import Any

from student_scripts.a2k.run_suite import attention_cases, correctness_cases
from student_scripts.a2k.runtime import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    MIN_FREE_MIB,
    STARTER_COMMIT,
    assert_public_payload,
)


SCHEMA_VERSION = "cs336.a2k.public-results.v1"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
WORKSPACE_ROOT = PROJECT_ROOT.parent
RUNTIME_ROOT = (PROJECT_ROOT / ".runtime").resolve()
REQUIRED_TASKS = {
    "unit_tests",
    "correctness",
    "checkpointing",
    "compile_comparison",
    "attention",
}
OFFICIAL_TEST_NODE_IDS = (
    "tests/test_attention.py::test_flash_forward_pass_pytorch",
    "tests/test_attention.py::test_flash_forward_pass_triton[False]",
    "tests/test_attention.py::test_flash_forward_pass_triton[True]",
    "tests/test_attention.py::test_flash_backward_pytorch",
    "tests/test_attention.py::test_flash_backward_triton[False]",
    "tests/test_attention.py::test_flash_backward_triton[True]",
)
SUCCESS_STATUSES = {"ok", "development_ok", "dry_run_ok", "pass", "development_pass"}
CHECKPOINT_COLUMNS = (
    "config_id",
    "model_size",
    "parameter_count",
    "parameter_dtype",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "seed",
    "allocator_limit_mib",
    "allocator_fraction",
    "within_allocator_limit",
    "error_type",
    "error_summary",
    "authoritative",
    "evidence_class",
)
ATTENTION_COLUMNS = (
    "implementation",
    "matrix",
    "batch_size",
    "sequence_length",
    "head_dim",
    "dtype",
    "causal",
    "phase",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "timer",
    "warmup_ms",
    "rep_ms",
    "quantiles",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "speedup_vs_eager",
    "query_tile",
    "key_tile",
    "num_warps",
    "num_stages",
    "status",
    "authoritative",
    "evidence_class",
)
COMPILE_COLUMNS = (
    "config_id",
    "workload",
    "model_size",
    "parameter_count",
    "parameter_dtype",
    "batch_size",
    "sequence_length",
    "context_length",
    "head_dim",
    "dtype",
    "causal",
    "implementation",
    "phase",
    "warmup_steps",
    "measurement_steps",
    "timer",
    "warmup_ms",
    "rep_ms",
    "measurement_count",
    "first_call_ms",
    "cold_start_ms",
    "cold_forward_setup_ms",
    "cold_phase_ms",
    "steady_ms_samples",
    "raw_samples_retained_privately",
    "steady_ms_p20",
    "steady_ms_p50",
    "steady_ms_p80",
    "cold_peak_allocated_mib",
    "cold_peak_reserved_mib",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "compile_backend",
    "compile_fullgraph",
    "compile_dynamic",
    "status",
    "seed",
    "allocator_limit_mib",
    "allocator_fraction",
    "within_allocator_limit",
    "error_type",
    "error_summary",
    "authoritative",
    "evidence_class",
)
EXPECTED_TRITON_CONFIG = {
    "block_m": 64,
    "block_n": 64,
    "num_warps": 4,
    "num_stages": 2,
}
EXPECTED_MODEL_PARAMETER_COUNTS = {
    "checkpoint_dry_run": 86_560,
    "checkpoint_medium": 423_183_360,
    "compile_dry_run": 24_736,
    "compile_small": 128_625_408,
}
_PRIVATE_KEY = re.compile(
    r"(?:password|passwd|secret|cookie|token|api.?key|private.?key|gpu.?uuid|hostname|host_name|"
    r"user_name|username|internal.?id|process.?list)",
    re.IGNORECASE,
)
_PRIVATE_STRING = re.compile(
    r"(?<![A-Za-z0-9])/(?:root|home|inspire|workspace|mnt|var|opt|tmp)(?:/|\b)|"
    r"\b(?:\d{1,3}\.){3}\d{1,3}\b|\b[0-9a-f]{8}-[0-9a-f-]{27,}\b",
    re.IGNORECASE,
)


class EvidenceValidationError(RuntimeError):
    """Raised before publication when raw evidence is incomplete or unsafe."""


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise EvidenceValidationError(message)


def _finite(value: Any, *, minimum: float | None = None) -> bool:
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        return False
    number = float(value)
    if not math.isfinite(number):
        return False
    return minimum is None or number >= minimum


def _positive(value: Any) -> bool:
    return _finite(value) and float(value) > 0


def _same_number(actual: Any, expected: float, *, tolerance: float = 1e-6) -> bool:
    return _finite(actual) and math.isclose(float(actual), float(expected), rel_tol=1e-7, abs_tol=tolerance)


def _linear_quantile(samples: Sequence[float], probability: float) -> float:
    ordered = sorted(float(value) for value in samples)
    _require(bool(ordered), "cannot validate a quantile from an empty sample set")
    position = (len(ordered) - 1) * probability
    lower = math.floor(position)
    upper = math.ceil(position)
    if lower == upper:
        return ordered[lower]
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def _validate_seed(value: Any, *, expected: int, label: str) -> None:
    _require(isinstance(value, int) and not isinstance(value, bool) and value >= 0, f"{label} seed is invalid")
    _require(value == expected, f"{label} seed does not match the suite seed")


def _validate_quantiles(container: Any, *, prefix: str, label: str) -> None:
    _require(isinstance(container, Mapping), f"{label} lacks latency quantiles")
    p20 = container.get(f"{prefix}p20_ms")
    p50 = container.get(f"{prefix}p50_ms")
    p80 = container.get(f"{prefix}p80_ms")
    _require(all(_positive(value) for value in (p20, p50, p80)), f"{label} quantiles must be finite and positive")
    _require(float(p20) <= float(p50) <= float(p80), f"{label} quantiles are not ordered")


def _validate_samples(
    samples: Any,
    *,
    expected_count: Any,
    p50: Any,
    label: str,
    p20: Any | None = None,
    p80: Any | None = None,
) -> None:
    _require(isinstance(expected_count, int) and not isinstance(expected_count, bool) and expected_count > 0, f"{label} sample count is invalid")
    _require(isinstance(samples, list) and len(samples) == expected_count, f"{label} raw latency samples are incomplete")
    _require(all(_positive(value) for value in samples), f"{label} latency sample is not finite and positive")
    _require(_same_number(p50, statistics.median(samples)), f"{label} p50 does not match the raw-sample median")
    if p20 is not None or p80 is not None:
        _require(_same_number(p20, _linear_quantile(samples, 0.2)), f"{label} p20 does not match the raw samples")
        _require(_same_number(p80, _linear_quantile(samples, 0.8)), f"{label} p80 does not match the raw samples")


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise EvidenceValidationError(f"missing or invalid raw JSON artifact: {path.name}") from exc
    if not isinstance(value, dict):
        raise EvidenceValidationError(f"raw JSON artifact is not an object: {path.name}")
    return value


def _within_workspace(path: Path, label: str, *, allow_runtime: bool = False) -> Path:
    resolved = path.expanduser().resolve()
    roots = [WORKSPACE_ROOT.resolve()]
    if allow_runtime:
        roots.append(RUNTIME_ROOT)
    for root in roots:
        try:
            resolved.relative_to(root)
            return resolved
        except ValueError:
            continue
    suffix = " or the project runtime root" if allow_runtime else ""
    raise EvidenceValidationError(f"{label} must stay below the shared cs336 workspace{suffix}")


def _validate_public_tree(value: Any, *, key: str | None = None) -> None:
    if key is not None and _PRIVATE_KEY.search(key):
        raise EvidenceValidationError(f"public payload contains forbidden private field: {key}")
    if isinstance(value, Mapping):
        for child_key, child_value in value.items():
            _validate_public_tree(child_value, key=str(child_key))
    elif isinstance(value, (list, tuple)):
        for child in value:
            _validate_public_tree(child)
    elif isinstance(value, str) and _PRIVATE_STRING.search(value):
        raise EvidenceValidationError("public payload contains an internal path, IP address, or UUID")


def _assert_public(value: Any) -> None:
    try:
        assert_public_payload(value)
    except Exception as exc:
        raise EvidenceValidationError("public payload failed path/IP/UUID screening") from exc
    _validate_public_tree(value)


def _evidence_class(mode: str) -> str:
    return {
        "formal_cuda": "formal_rtx4090_24gb",
        "development_cuda": "development_non_authoritative",
        "dry_run": "cpu_dry_run_non_authoritative",
    }[mode]


def _non_authoritative_reason(mode: str, gpu_name: str | None) -> str | None:
    if mode == "formal_cuda":
        return None
    if mode == "development_cuda":
        label = gpu_name or "a non-standard CUDA GPU"
        return f"Development result on {label} under a 23552 MiB PyTorch allocator cap; it does not replace the required RTX 4090 24GB formal evidence."
    return "CPU dry-run; not valid GPU correctness, performance, or allocator evidence."


def _validate_manifest(raw_dir: Path) -> dict[str, Any]:
    manifest = _load_json(raw_dir / "manifest.json")
    _require(manifest.get("schema_version") == "cs336.a2k.suite-manifest.v1", "unexpected suite manifest schema")
    _require(manifest.get("status") == "completed", "suite did not complete successfully")
    _require(manifest.get("starter_commit") == STARTER_COMMIT, "starter commit mismatch")
    _require(
        isinstance(manifest.get("seed"), int) and not isinstance(manifest.get("seed"), bool) and manifest["seed"] >= 0,
        "suite seed is missing or invalid",
    )
    mode = manifest.get("execution_mode")
    _require(mode in {"formal_cuda", "development_cuda", "dry_run"}, "unknown execution mode")
    _require(manifest.get("authoritative") is (mode == "formal_cuda"), "manifest authority classification mismatch")
    _require(manifest.get("serial_execution") is True, "suite was not marked serial")
    tasks = manifest.get("tasks")
    _require(isinstance(tasks, dict) and set(tasks) == REQUIRED_TASKS, "suite manifest task set is incomplete")
    for name, task in tasks.items():
        _require(isinstance(task, dict) and task.get("state") == "completed", f"suite task is incomplete: {name}")
        _require(isinstance(task.get("return_codes"), list), f"suite task lacks return-code evidence: {name}")
    return manifest


def _validate_runtime(runtime: Any, *, mode: str, label: str) -> dict[str, Any]:
    _require(isinstance(runtime, dict), f"{label} lacks runtime metadata")
    expected_authority = mode == "formal_cuda"
    _require(runtime.get("authoritative") is expected_authority, f"{label} authority classification mismatch")
    allocator = runtime.get("allocator")
    _require(isinstance(allocator, dict), f"{label} lacks allocator evidence")
    _require(allocator.get("allocator_limit_mib") == ALLOCATOR_LIMIT_MIB, f"{label} used the wrong allocator limit")
    if mode == "dry_run":
        _require(runtime.get("device_type") == "cpu", f"{label} dry-run is not CPU-classified")
        _require(allocator.get("guard_applied") is False, f"{label} falsely claims a CUDA allocator guard")
        return runtime

    _require(runtime.get("device_type") == "cuda", f"{label} is not CUDA evidence")
    _require(allocator.get("guard_applied") is True, f"{label} allocator guard was not applied")
    _require(allocator.get("applied_before_first_cuda_allocation") is True, f"{label} allocator guard ordering is missing")
    _require(allocator.get("prior_allocated_bytes") == 0, f"{label} allocated CUDA memory before the guard")
    fraction = allocator.get("allocator_fraction")
    _require(_finite(fraction, minimum=0.0) and 0 < float(fraction) <= 1, f"{label} allocator fraction is invalid")
    gpu = runtime.get("gpu")
    _require(isinstance(gpu, dict), f"{label} lacks public GPU metadata")
    name = gpu.get("name")
    total = gpu.get("memory_total_mib")
    free = gpu.get("memory_free_mib")
    _require(isinstance(name, str) and name.strip(), f"{label} GPU name is missing")
    _require(_finite(total, minimum=1), f"{label} total GPU memory is invalid")
    _require(_finite(free, minimum=MIN_FREE_MIB), f"{label} started below 22 GiB free memory")
    if mode == "formal_cuda":
        _require(" ".join(name.split()) == "NVIDIA GeForce RTX 4090", f"{label} is not RTX 4090 evidence")
        _require(24_000 <= float(total) <= 25_000, f"{label} is not a 24GB 4090")
        _require(runtime.get("execution_mode") == "formal_cuda", f"{label} does not declare formal CUDA mode")
    else:
        _require(runtime.get("execution_mode") == "development_cuda", f"{label} does not declare development CUDA mode")
        _require(isinstance(runtime.get("non_authoritative_reason"), str), f"{label} lacks a non-authoritative reason")
    return runtime


def _validate_memory_pair(record: Mapping[str, Any], *, label: str, required: bool = False) -> None:
    allocated = record.get("peak_allocated_mib")
    reserved = record.get("peak_reserved_mib")
    if allocated is None and reserved is None:
        _require(not required, f"{label} lacks required peak-memory evidence")
        return
    _require(allocated is not None and reserved is not None, f"{label} has only one peak-memory field")
    _require(_finite(allocated, minimum=0) and _finite(reserved, minimum=0), f"{label} has invalid peak memory")
    _require(float(allocated) <= float(reserved) + 1e-6, f"{label} allocated peak exceeds reserved peak")
    _require(float(reserved) <= ALLOCATOR_LIMIT_MIB + 1e-6, f"{label} exceeds the 23 GiB allocator limit")


def _validate_allocator_row(record: Mapping[str, Any], runtime: Mapping[str, Any], *, mode: str, label: str) -> None:
    allocator = runtime["allocator"]
    _require(record.get("allocator_limit_mib") == ALLOCATOR_LIMIT_MIB, f"{label} allocator limit is missing or incorrect")
    _require(record.get("allocator_fraction") == allocator.get("allocator_fraction"), f"{label} allocator fraction disagrees with runtime metadata")
    if mode != "dry_run" and record.get("status") == "ok":
        _require(record.get("within_allocator_limit") is True, f"{label} is not marked within the allocator limit")


def _validate_attention_outcome(
    record: Mapping[str, Any],
    *,
    mode: str,
    required_success: bool,
    label: str,
) -> None:
    status = record.get("status")
    if mode == "dry_run":
        expected_status = "dry_run_skipped" if record.get("implementation") == "triton" else "dry_run_ok"
        _require(status == expected_status, f"{label} CPU dry-run status mismatch")
    else:
        success_status = "ok" if mode == "formal_cuda" else "development_ok"
        _require(
            status == success_status or (not required_success and status == "oom"),
            f"{label} is neither a required success nor an allowed boundary eager OOM",
        )
    if status in SUCCESS_STATUSES:
        _validate_quantiles(record.get("latency"), prefix="", label=label)
        _require(record.get("error") is None, f"{label} successful row contains an error")
    elif status == "oom":
        error = record.get("error")
        _require(
            isinstance(error, dict) and error.get("category") == "out_of_memory" and error.get("message") == "allocator OOM; case retained without fallback",
            f"{label} OOM category is missing or dishonest",
        )
        _require(not record.get("latency"), f"{label} OOM row contains latency measurements")


def _load_attention(
    raw_dir: Path,
    *,
    mode: str,
    expected_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    index = _load_json(raw_dir / "attention" / "index.json")
    _require(index.get("schema_version") == "cs336.a2k.attention-index.v1", "attention index schema mismatch")
    _require(index.get("mode") == mode, "attention index execution mode mismatch")
    _require(index.get("one_fresh_python_process_per_case") is True, "attention cases were not process-isolated")
    _require(index.get("serial") is True, "attention cases were not serial")
    expected = attention_cases(dry_run=mode == "dry_run")
    listed = index.get("cases")
    _require(isinstance(listed, list) and len(listed) == len(expected), "attention index has the wrong case count")
    _require(index.get("expected_case_count") == len(expected), "attention index expected-case count mismatch")
    expected_by_id = {case["case_id"]: case for case in expected}
    _require(len(expected_by_id) == len(expected), "internal attention matrix contains duplicate case IDs")
    observed_ids: set[str] = set()
    for item in listed:
        _require(isinstance(item, dict), "attention index contains a non-object case")
        case_id = item.get("case_id")
        _require(isinstance(case_id, str) and case_id in expected_by_id, "attention index contains an unknown case ID")
        _require(case_id not in observed_ids, "attention index contains a duplicate case ID")
        observed_ids.add(case_id)
        expected_item = expected_by_id[case_id]
        for field in ("matrix", "sequence_length", "head_dim", "implementation", "phase"):
            _require(item.get(field) == expected_item[field], f"attention index field mismatch: {case_id}/{field}")
        _require(item.get("artifact_present") is True, f"attention artifact is missing: {case_id}")
        _require(item.get("return_code") == 0, f"attention worker returned a nonzero status: {case_id}")
    _require(observed_ids == set(expected_by_id), "attention index does not match the fixed matrix")

    records: list[dict[str, Any]] = []
    runtimes: list[dict[str, Any]] = []
    for item in listed:
        case_id = item.get("case_id")
        _require(isinstance(case_id, str) and re.fullmatch(r"[A-Za-z0-9_.-]+", case_id) is not None, "unsafe attention case id")
        record = _load_json(raw_dir / "attention" / "cases" / f"{case_id}.json")
        _require(record.get("schema_version") == "cs336.a2k.attention-benchmark-case.v1", f"attention case schema mismatch: {case_id}")
        _require(record.get("case_id") == case_id, f"attention case ID mismatch: {case_id}")
        for field in ("sequence_length", "head_dim", "implementation", "phase"):
            _require(record.get(field) == item.get(field), f"attention case identity mismatch: {case_id}")
        expected_record_matrix = "dry_run" if mode == "dry_run" else item["matrix"]
        _require(record.get("matrix") == expected_record_matrix, f"attention matrix classification mismatch: {case_id}")
        _require(record.get("batch_size") == 1, f"attention batch size mismatch: {case_id}")
        _require(record.get("causal") is True, f"attention case is not causal: {case_id}")
        _require(record.get("dtype") == ("float32" if mode == "dry_run" else "bfloat16"), f"attention dtype mismatch: {case_id}")
        _validate_seed(record.get("seed"), expected=expected_seed, label=f"attention {case_id}")
        _require(record.get("fallback_used") is False, f"attention case used or omitted the no-fallback marker: {case_id}")
        _require(record.get("input_creation_timed") is False, f"attention case included input creation in timing: {case_id}")
        timer = record.get("timer")
        _require(isinstance(timer, dict), f"attention case lacks timer metadata: {case_id}")
        _require(timer.get("quantiles") == [0.2, 0.5, 0.8], f"attention quantiles mismatch: {case_id}")
        if mode != "dry_run":
            _require(timer.get("name") == "triton.testing.do_bench", f"attention timer mismatch: {case_id}")
            _require(timer.get("warmup_ms") == 100 and timer.get("rep_ms") == 300, f"attention timing protocol mismatch: {case_id}")
            _require(timer.get("synchronize_boundaries") is True, f"attention synchronization marker is missing: {case_id}")
        else:
            _require(timer.get("name") == "time.perf_counter", f"attention dry-run timer mismatch: {case_id}")

        if record.get("implementation") == "triton":
            _require(record.get("triton_config") == EXPECTED_TRITON_CONFIG, f"Triton launch config mismatch: {case_id}")
        else:
            _require(record.get("triton_config") is None, f"non-Triton attention row contains a Triton config: {case_id}")
        if record.get("implementation") == "compiled":
            compile_metadata = record.get("compile")
            _require(isinstance(compile_metadata, dict), f"compiled attention metadata is missing: {case_id}")
            _require(
                compile_metadata.get("backend") == ("eager" if mode == "dry_run" else "inductor")
                and compile_metadata.get("fullgraph") is (mode != "dry_run")
                and compile_metadata.get("dynamic") is False,
                f"compiled attention configuration mismatch: {case_id}",
            )
        runtime = _validate_runtime(record.get("runtime"), mode=mode, label=f"attention {case_id}")
        _require(record.get("authoritative") is (mode == "formal_cuda"), f"attention authority mismatch: {case_id}")
        runtimes.append(runtime)
        memory = record.get("memory")
        if isinstance(memory, dict):
            _validate_memory_pair(memory, label=f"attention {case_id}", required=mode != "dry_run")
        else:
            _require(
                mode == "dry_run" and record.get("status") == "dry_run_skipped",
                f"attention case lacks memory evidence: {case_id}",
            )
        _validate_attention_outcome(
            record,
            mode=mode,
            required_success=record.get("matrix") == "core" or record.get("implementation") == "triton",
            label=f"attention {case_id}",
        )
        records.append(record)
    return records, runtimes


def _validate_checkpointing(
    payload: dict[str, Any],
    *,
    mode: str,
    expected_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(payload.get("schema_version") == 1, "checkpoint artifact schema mismatch")
    _require(payload.get("benchmark") == "activation_checkpointing", "checkpoint benchmark identity mismatch")
    _require(payload.get("formal_evidence") is (mode == "formal_cuda"), "checkpoint authority classification mismatch")
    _require("fresh Python process" in str(payload.get("process_isolation")), "checkpoint cases were not process-isolated")
    contract = payload.get("measurement_contract")
    _require(isinstance(contract, dict), "checkpoint measurement contract is missing")
    _require(contract.get("model_parameters") == "fp32", "checkpoint parameters were not declared FP32")
    _require(
        contract.get("autocast") == ("disabled_cpu_dry_run" if mode == "dry_run" else "bf16"),
        "checkpoint autocast contract mismatch",
    )
    _require(contract.get("peak_reset_after_warmup") is True, "checkpoint peak-reset contract is missing")
    rows = payload.get("results")
    _require(isinstance(rows, list) and len(rows) == 7, "checkpoint matrix must contain seven rows")
    small_context, large_context = (8, 16) if mode == "dry_run" else (1024, 2048)
    small = [row for row in rows if row.get("context_length") == small_context]
    large = [row for row in rows if row.get("context_length") == large_context]
    _require({row.get("checkpoint_block_size") for row in small} == {0, 1, 2, 4, 8}, "checkpoint 1024 matrix is incomplete")
    _require(len(large) == 2 and {row.get("checkpoint_block_size") for row in large if row.get("checkpoint_block_size") == 0} == {0}, "checkpoint 2048 boundary is incomplete")
    selected = payload.get("selection", {}).get("context_2048_checkpoint_block_size")
    _require(isinstance(selected, int) and selected in {1, 2, 4, 8}, "checkpoint boundary selection is invalid")
    _require(
        payload.get("selection", {}).get("criterion") == "lowest successful context-1024 checkpointed peak_allocated_mib",
        "checkpoint selection criterion mismatch",
    )
    _require({row.get("checkpoint_block_size") for row in large} == {0, selected}, "checkpoint boundary does not use the selected minimum-peak configuration")
    expected_coordinates = {(small_context, block) for block in (0, 1, 2, 4, 8)} | {
        (large_context, 0),
        (large_context, selected),
    }
    observed_coordinates = {(row.get("context_length"), row.get("checkpoint_block_size")) for row in rows}
    _require(observed_coordinates == expected_coordinates and len(observed_coordinates) == len(rows), "checkpoint rows contain duplicates or unexpected configurations")
    selected_boundary = next(row for row in large if row.get("checkpoint_block_size") == selected)
    _require(selected_boundary.get("status") == "ok", "selected context-2048 checkpoint configuration did not succeed")
    if mode != "dry_run":
        _require(all(row.get("status") == "ok" for row in small), "checkpoint context-1024 standard matrix did not fully succeed")
        checkpointed = [row for row in small if row.get("checkpoint_block_size") in {1, 2, 4, 8}]
        _require(
            all(_finite(row.get("peak_allocated_mib"), minimum=0) for row in checkpointed),
            "checkpoint selection lacks numeric context-1024 peak memory",
        )
        measured_minimum = min(checkpointed, key=lambda row: (float(row["peak_allocated_mib"]), row["checkpoint_block_size"]))
        _require(
            measured_minimum.get("checkpoint_block_size") == selected,
            "context-2048 checkpoint row is not the measured lowest-peak context-1024 configuration",
        )
    runtimes: list[dict[str, Any]] = []
    for row in rows:
        label = f"checkpoint {row.get('config_id')}"
        expected_model = "dry_run" if mode == "dry_run" else "medium"
        expected_layers = 8 if mode == "dry_run" else 24
        expected_dtype = "fp32" if mode == "dry_run" else "bf16"
        expected_count = EXPECTED_MODEL_PARAMETER_COUNTS[f"checkpoint_{expected_model}"]
        expected_id = f"{expected_model}_ctx{row.get('context_length')}_ckpt{row.get('checkpoint_block_size')}"
        _require(row.get("config_id") == expected_id, f"{label} configuration ID mismatch")
        _require(row.get("batch_size") == 1, "checkpoint batch size mismatch")
        _require(row.get("nested") is False, "fixed checkpoint experiment must be non-nested")
        _require(row.get("model_size") == expected_model and row.get("num_layers") == expected_layers, f"{label} model configuration mismatch")
        _require(row.get("dtype") == expected_dtype, f"{label} dtype mismatch")
        _require(row.get("parameter_count") == expected_count, f"{label} parameter count mismatch")
        _require(row.get("parameter_dtype") == "fp32", f"{label} parameters are not FP32")
        _validate_seed(row.get("seed"), expected=expected_seed, label=label)
        if mode == "dry_run":
            _require(row.get("warmup_steps") == 1 and row.get("measurement_steps") == 1, "checkpoint dry-run protocol mismatch")
        else:
            _require(row.get("warmup_steps", 0) >= 3 and row.get("measurement_steps", 0) >= 5, "checkpoint measurement protocol mismatch")
        runtime = _validate_runtime(row.get("runtime"), mode=mode, label=label)
        runtimes.append(runtime)
        _validate_allocator_row(row, runtime, mode=mode, label=label)
        _validate_memory_pair(row, label=label, required=mode != "dry_run")
        if row.get("status") == "ok":
            _validate_samples(
                row.get("step_time_ms_samples"),
                expected_count=row.get("measurement_steps"),
                p50=row.get("step_time_ms_p50"),
                label=label,
            )
            _require(row.get("error_type") is None and row.get("error_summary") is None, f"{label} successful row contains an error")
        elif row.get("status") == "oom":
            _require(
                row.get("context_length") == large_context
                and row.get("checkpoint_block_size") == 0
                and isinstance(row.get("error_type"), str)
                and row.get("error_summary") == "allocator OOM; case retained without fallback",
                f"{label} is not an allowed honest context-boundary OOM",
            )
            _require(not row.get("step_time_ms_samples") and row.get("step_time_ms_p50") is None, f"{label} OOM row contains latency data")
        else:
            raise EvidenceValidationError(f"{label} has an unsupported status")
    return rows, runtimes


def _expected_compile_keys(*, dry_run: bool) -> set[tuple[Any, ...]]:
    attention_shapes = ((8, 4),) if dry_run else ((512, 64), (2048, 128), (8192, 128))
    keys = {
        ("attention", sequence, head_dim, implementation, phase)
        for sequence, head_dim in attention_shapes
        for implementation in ("eager", "compiled")
        for phase in ("forward", "backward", "forward_backward")
    }
    model_size = "dry_run" if dry_run else "small"
    keys.update(("transformer", model_size, implementation, phase) for implementation in ("eager", "compiled") for phase in ("forward", "forward_backward", "training_step"))
    return keys


def _validate_compile_row_configuration(row: Mapping[str, Any], *, mode: str) -> tuple[Any, ...]:
    workload = row.get("workload")
    implementation = row.get("implementation")
    phase = row.get("phase")
    _require(implementation in {"eager", "compiled"}, "compile row implementation is invalid")
    _require(row.get("batch_size") == 1 and row.get("causal") is True, "compile comparison configuration mismatch")
    _require(row.get("dtype") == ("fp32" if mode == "dry_run" else "bf16"), "compile comparison dtype mismatch")
    if workload == "attention":
        _require(row.get("model_size") is None and row.get("context_length") is None, "compile attention row contains model fields")
        _require(row.get("parameter_count") is None and row.get("parameter_dtype") is None, "compile attention row contains model parameters")
        sequence = row.get("sequence_length")
        head_dim = row.get("head_dim")
        expected_id = f"attention_s{sequence}_d{head_dim}_{implementation}_{phase}"
        _require(row.get("config_id") == expected_id, "compile attention configuration ID mismatch")
        key = ("attention", sequence, head_dim, implementation, phase)
    elif workload == "transformer":
        expected_model = "dry_run" if mode == "dry_run" else "small"
        expected_context = 8 if mode == "dry_run" else 512
        expected_count = EXPECTED_MODEL_PARAMETER_COUNTS[f"compile_{expected_model}"]
        _require(row.get("model_size") == expected_model, "compile transformer model size mismatch")
        _require(row.get("context_length") == expected_context, "compile transformer context length mismatch")
        _require(row.get("sequence_length") is None and row.get("head_dim") is None, "compile transformer row contains attention shape fields")
        _require(row.get("parameter_count") == expected_count, "compile transformer parameter count mismatch")
        _require(row.get("parameter_dtype") == "fp32", "compile transformer parameters are not FP32")
        expected_id = f"model_{expected_model}_ctx{expected_context}_{implementation}_{phase}"
        _require(row.get("config_id") == expected_id, "compile transformer configuration ID mismatch")
        key = ("transformer", expected_model, implementation, phase)
    else:
        raise EvidenceValidationError("compile row workload is invalid")

    if implementation == "compiled":
        _require(
            row.get("compile_backend") == ("eager" if mode == "dry_run" else "inductor") and row.get("compile_fullgraph") is True and row.get("compile_dynamic") is False,
            "compiled row must use the fixed backend/fullgraph/dynamic configuration",
        )
    else:
        _require(
            row.get("compile_backend") is None and row.get("compile_fullgraph") is False and row.get("compile_dynamic") is False,
            "eager row contains compiled configuration metadata",
        )
    return key


def _validate_compile(
    payload: dict[str, Any],
    *,
    mode: str,
    expected_seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    _require(payload.get("schema_version") == 1, "compile artifact schema mismatch")
    _require(payload.get("benchmark") == "torch_compile_comparison", "compile benchmark identity mismatch")
    _require(payload.get("formal_evidence") is (mode == "formal_cuda"), "compile authority classification mismatch")
    _require("fresh Python process" in str(payload.get("process_isolation")), "compile cases were not process-isolated")
    contract = payload.get("measurement_contract")
    _require(isinstance(contract, dict), "compile measurement contract is missing")
    _require(contract.get("first_compiled_execution_reported_as_cold_start") is True, "compile cold-start contract is missing")
    _require(contract.get("cold_start_excluded_from_warmup_and_steady_samples") is True, "compile cold-start leaked into steady samples")
    _require(contract.get("compile_dynamic") is False, "compile dynamic-shape contract mismatch")
    _require(
        contract.get("dtype") == ("fp32_cpu_dry_run" if mode == "dry_run" else "bf16"),
        "compile dtype contract mismatch",
    )
    rows = payload.get("results")
    expected = _expected_compile_keys(dry_run=mode == "dry_run")
    _require(isinstance(rows, list) and len(rows) == len(expected), "compile matrix has the wrong row count")
    observed: set[tuple[Any, ...]] = set()
    runtimes: list[dict[str, Any]] = []
    for row in rows:
        _require(isinstance(row, dict), "compile matrix contains a non-object row")
        key = _validate_compile_row_configuration(row, mode=mode)
        _require(key not in observed, "compile matrix contains a duplicate row")
        observed.add(key)
        label = f"compile {row.get('config_id')}"
        _validate_seed(row.get("seed"), expected=expected_seed, label=label)
        if mode == "dry_run":
            _require(row.get("warmup_steps") == 1 and row.get("measurement_steps") == 1, "compile dry-run protocol mismatch")
        elif row.get("workload") == "attention":
            _require(
                row.get("timer") == "cuda_event_duration" and row.get("warmup_ms") == 100 and row.get("rep_ms") == 300,
                "compile attention timing protocol mismatch",
            )
            _require(row.get("warmup_steps") is None and row.get("measurement_steps") is None, "compile attention row mixed step and duration protocols")
        else:
            _require(row.get("warmup_steps", 0) >= 3 and row.get("measurement_steps", 0) >= 5, "compile model measurement protocol mismatch")
            _require(
                row.get("timer") == "synchronized_perf_counter" and row.get("warmup_ms") is None and row.get("rep_ms") is None,
                "compile transformer timing protocol mismatch",
            )
        runtime = _validate_runtime(row.get("runtime"), mode=mode, label=label)
        runtimes.append(runtime)
        _validate_allocator_row(row, runtime, mode=mode, label=label)
        _validate_memory_pair(row, label=label, required=mode != "dry_run")
        cold_memory = {
            "peak_allocated_mib": row.get("cold_peak_allocated_mib"),
            "peak_reserved_mib": row.get("cold_peak_reserved_mib"),
        }
        _validate_memory_pair(cold_memory, label=f"{label} cold start", required=mode != "dry_run")
        if row.get("status") == "ok":
            samples = row.get("steady_ms_samples")
            expected_samples = row.get("measurement_count")
            if row.get("workload") == "transformer" or mode == "dry_run":
                _require(expected_samples == row.get("measurement_steps"), f"{label} measurement count mismatch")
            _validate_samples(
                samples,
                expected_count=expected_samples,
                p20=row.get("steady_ms_p20"),
                p50=row.get("steady_ms_p50"),
                p80=row.get("steady_ms_p80"),
                label=label,
            )
            _require(_positive(row.get("first_call_ms")) and _positive(row.get("cold_phase_ms")), f"{label} first-call timing is missing")
            if row.get("phase") == "backward":
                _require(_positive(row.get("cold_forward_setup_ms")), f"{label} backward cold setup timing is missing")
            else:
                _require(row.get("cold_forward_setup_ms") is None, f"{label} non-backward row contains backward setup timing")
            if row.get("implementation") == "compiled":
                _require(
                    _positive(row.get("cold_start_ms")) and _same_number(row.get("cold_start_ms"), float(row["first_call_ms"])),
                    f"{label} compiled cold-start time is missing or inconsistent",
                )
            else:
                _require(row.get("cold_start_ms") is None, f"{label} eager row contains a compile cold-start time")
            _require(row.get("error_type") is None and row.get("error_summary") is None, f"{label} successful row contains an error")
    _require(observed == expected, "compile matrix does not match the fixed configurations")
    _require(all(row.get("status") == "ok" for row in rows), "eager/compiled comparison contains a failed or OOM row")
    return rows, runtimes


def _correctness_case_id(case: Mapping[str, Any]) -> str:
    causal = "causal" if case["causal"] else "noncausal"
    return f"seed{case['seed']}-n{case['sequence_length']}-d{case['head_dim']}-{case['dtype']}-{causal}"


def _validate_correctness_metrics(
    metrics: Any,
    *,
    dtype: str,
    sequence_length: int,
    head_dim: int,
    label: str,
) -> None:
    _require(isinstance(metrics, dict) and set(metrics) == {"O", "L", "dQ", "dK", "dV"}, f"{label} O/L/dQ/dK/dV metrics are incomplete")
    expected_tolerance = (0.04, 0.05) if dtype == "bfloat16" else (0.0002, 0.0002)
    expected_elements = {
        "O": sequence_length * head_dim,
        "L": sequence_length,
        "dQ": sequence_length * head_dim,
        "dK": sequence_length * head_dim,
        "dV": sequence_length * head_dim,
    }
    for name, metric in metrics.items():
        _require(
            isinstance(metric, dict)
            and _finite(metric.get("max_absolute_error"), minimum=0)
            and _finite(metric.get("max_relative_error"), minimum=0)
            and _same_number(metric.get("atol"), expected_tolerance[0])
            and _same_number(metric.get("rtol"), expected_tolerance[1])
            and metric.get("element_count") == expected_elements[name]
            and metric.get("passed") is True,
            f"{label} {name} metric is malformed or failed",
        )


def _validate_correctness(payload: dict[str, Any], *, mode: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    _require(payload.get("schema_version") == "cs336.a2k.correctness.v1", "correctness artifact schema mismatch")
    _require(payload.get("reference") == "explicit FP32 QK^T / scale / mask / softmax / PV", "correctness reference identity mismatch")
    _require(payload.get("tf32_required") == "ieee", "correctness TF32 contract mismatch")
    _require(payload.get("authoritative") is (mode == "formal_cuda"), "correctness authority classification mismatch")
    cases = payload.get("cases")
    expected_cases = correctness_cases(dry_run=mode == "dry_run")
    expected_count = len(expected_cases)
    _require(isinstance(cases, list) and len(cases) == expected_count, "correctness matrix has the wrong case count")
    runtime = _validate_runtime(payload.get("runtime"), mode=mode, label="correctness")
    runtime_evidence = payload.get("case_runtime_evidence")
    _require(
        isinstance(runtime_evidence, list) and len(runtime_evidence) == expected_count,
        "correctness cases were not independently process-isolated",
    )
    expected_by_id = {_correctness_case_id(case): case for case in expected_cases}
    _require(len(expected_by_id) == expected_count, "internal correctness matrix contains duplicate case IDs")
    evidence_ids: set[str] = set()
    case_runtimes = [runtime]
    for evidence in runtime_evidence:
        _require(isinstance(evidence, dict), "correctness runtime evidence is malformed")
        case_id = evidence.get("case_id")
        _require(isinstance(case_id, str) and case_id in expected_by_id and case_id not in evidence_ids, "correctness runtime evidence case ID mismatch")
        evidence_ids.add(case_id)
        case_runtimes.append(_validate_runtime(evidence.get("runtime"), mode=mode, label=f"correctness {case_id}"))
    _require(evidence_ids == set(expected_by_id), "correctness runtime evidence matrix is incomplete")

    observed_ids: set[str] = set()
    for case in cases:
        _require(isinstance(case, dict), "correctness matrix contains a non-object case")
        case_id = case.get("case_id")
        _require(isinstance(case_id, str) and case_id in expected_by_id and case_id not in observed_ids, "correctness case identity is invalid or duplicated")
        observed_ids.add(case_id)
        expected_case = expected_by_id[case_id]
        for field in ("seed", "sequence_length", "head_dim", "causal", "dtype"):
            _require(case.get(field) == expected_case[field], f"correctness case field mismatch: {case_id}/{field}")
        _require(case.get("batch_size") == 1, f"correctness batch size mismatch: {case_id}")
        _require(case.get("reference_compute_dtype") == "float32", f"correctness reference dtype mismatch: {case_id}")
        _require(case.get("tf32_policy") == "ieee", f"correctness TF32 policy mismatch: {case_id}")
        _require(case.get("status") == "pass", f"correctness case did not pass: {case_id}")
        implementations = case.get("implementations")
        _require(isinstance(implementations, dict) and set(implementations) == {"pytorch_tiled", "triton"}, "correctness implementation pair is incomplete")
        for implementation, result in implementations.items():
            _require(isinstance(result, dict) and isinstance(result.get("status"), str), "correctness result status is missing")
            if result.get("status") in {"pass", "fail"}:
                _validate_correctness_metrics(
                    result.get("metrics"),
                    dtype=case["dtype"],
                    sequence_length=case["sequence_length"],
                    head_dim=case["head_dim"],
                    label=f"correctness {case_id}/{implementation}",
                )
            memory = result.get("memory")
            if isinstance(memory, dict):
                _validate_memory_pair(memory, label=f"correctness {case_id}/{implementation}", required=mode != "dry_run")
            elif mode != "dry_run":
                raise EvidenceValidationError(f"correctness {case_id}/{implementation} lacks peak-memory evidence")
        if mode == "dry_run":
            _require(implementations["pytorch_tiled"].get("status") == "pass", "CPU tiled correctness dry-run failed")
            _require(
                implementations["triton"].get("status") == "skipped_non_authoritative",
                "CPU dry-run must not claim a Triton CUDA pass",
            )
            _require(case.get("pytorch_tiled_vs_triton") is None, "CPU dry-run contains a false Triton cross-check")
        else:
            _require(
                implementations["pytorch_tiled"].get("status") == "pass" and implementations["triton"].get("status") == "pass",
                "a required PyTorch-tiled or Triton correctness check failed",
            )
            for result in implementations.values():
                _require(all(metric.get("passed") is True for metric in result["metrics"].values()), "a required O/L/dQ/dK/dV metric failed")
            cross = case.get("pytorch_tiled_vs_triton")
            _require(isinstance(cross, dict) and cross.get("status") == "pass", "PyTorch-tiled versus Triton cross-check failed")
            _validate_correctness_metrics(
                cross.get("metrics"),
                dtype=case["dtype"],
                sequence_length=case["sequence_length"],
                head_dim=case["head_dim"],
                label=f"correctness {case_id}/cross-check",
            )
    _require(observed_ids == set(expected_by_id), "correctness case matrix is incomplete")
    summary = payload.get("summary")
    expected_checks = 1 if mode == "dry_run" else 38
    _require(
        isinstance(summary, dict)
        and summary.get("case_count") == expected_count
        and summary.get("implementation_check_count") == expected_checks
        and summary.get("passed") == expected_checks
        and summary.get("failed") == 0
        and summary.get("errors") == 0
        and summary.get("oom") == 0
        and summary.get("skipped") == (1 if mode == "dry_run" else 0)
        and summary.get("bf16_case_count") == (0 if mode == "dry_run" else 18)
        and summary.get("fp32_tf32_disabled_case_count") == 1,
        "correctness summary does not match the validated matrix",
    )
    _require(
        payload.get("status") == ("dry_run_ok" if mode == "dry_run" else "pass" if mode == "formal_cuda" else "development_pass"),
        "correctness aggregate status mismatch",
    )
    return payload, case_runtimes


def _validate_unit_tests(payload: dict[str, Any], *, mode: str) -> dict[str, Any]:
    _require(payload.get("schema_version") == "cs336.a2k.unit-tests.v1", "unit-test artifact schema mismatch")
    _require(payload.get("command") == "uv run pytest tests/test_attention.py -v", "official unit-test command mismatch")
    summary = payload.get("summary")
    _require(isinstance(summary, dict) and summary.get("parsed") is True, "unit-test summary could not be parsed")
    _require(isinstance(summary.get("total"), int) and summary["total"] > 0, "unit-test count is missing")
    _require(payload.get("return_code") == 0, "official unit tests returned a nonzero status")
    _require(summary.get("failed") == 0 and summary.get("errors") == 0, "official unit tests contain failures or errors")
    if mode != "dry_run":
        _require(
            summary.get("total") == 6 and summary.get("passed") == 6 and summary.get("skipped") == 0 and summary.get("xfailed") == 0 and summary.get("xpassed") == 0,
            "GPU unit tests must report exactly 6 passed with no other outcomes",
        )
    return payload


def _consistent_runtime(runtimes: list[dict[str, Any]], *, mode: str) -> dict[str, Any]:
    _require(runtimes, "no runtime metadata was found")
    first = runtimes[0]
    if mode == "dry_run":
        return first
    first_gpu = first["gpu"]
    first_software = first.get("software")
    for runtime in runtimes[1:]:
        gpu = runtime["gpu"]
        _require(gpu.get("name") == first_gpu.get("name"), "GPU model changed across processes")
        _require(abs(float(gpu.get("memory_total_mib")) - float(first_gpu.get("memory_total_mib"))) <= 64, "GPU total memory changed across processes")
        _require(gpu.get("driver_version") == first_gpu.get("driver_version"), "driver changed across processes")
        _require(runtime.get("software") == first_software, "software versions changed across processes")
    return first


def _attention_public_rows(records: list[dict[str, Any]], *, mode: str) -> list[dict[str, Any]]:
    evidence = _evidence_class(mode)
    authoritative = mode == "formal_cuda"
    rows: list[dict[str, Any]] = []
    eager: dict[tuple[int, int, str], float] = {}
    for record in records:
        latency = record.get("latency") if isinstance(record.get("latency"), dict) else {}
        if record.get("implementation") == "eager" and record.get("status") in SUCCESS_STATUSES and _finite(latency.get("p50_ms"), minimum=0):
            eager[(record["sequence_length"], record["head_dim"], record["phase"])] = float(latency["p50_ms"])
    for record in records:
        latency = record.get("latency") if isinstance(record.get("latency"), dict) else {}
        memory = record.get("memory") if isinstance(record.get("memory"), dict) else {}
        timer = record.get("timer") if isinstance(record.get("timer"), dict) else {}
        config = record.get("triton_config") if isinstance(record.get("triton_config"), dict) else {}
        p50 = latency.get("p50_ms")
        baseline = eager.get((record["sequence_length"], record["head_dim"], record["phase"]))
        speedup = None
        if record.get("status") in SUCCESS_STATUSES and _finite(p50, minimum=0) and float(p50) > 0 and baseline is not None:
            speedup = baseline / float(p50)
        rows.append(
            {
                "implementation": record.get("implementation"),
                "matrix": record.get("matrix"),
                "batch_size": record.get("batch_size"),
                "sequence_length": record.get("sequence_length"),
                "head_dim": record.get("head_dim"),
                "dtype": record.get("dtype"),
                "causal": record.get("causal"),
                "phase": record.get("phase"),
                "p20_ms": latency.get("p20_ms"),
                "p50_ms": p50,
                "p80_ms": latency.get("p80_ms"),
                "timer": timer.get("name"),
                "warmup_ms": timer.get("warmup_ms"),
                "rep_ms": timer.get("rep_ms"),
                "quantiles": timer.get("quantiles"),
                "peak_allocated_mib": memory.get("peak_allocated_mib"),
                "peak_reserved_mib": memory.get("peak_reserved_mib"),
                "speedup_vs_eager": round(speedup, 6) if speedup is not None else None,
                "query_tile": config.get("block_m"),
                "key_tile": config.get("block_n"),
                "num_warps": config.get("num_warps"),
                "num_stages": config.get("num_stages"),
                "status": record.get("status"),
                "authoritative": authoritative,
                "evidence_class": evidence,
            }
        )
    return rows


def _memory_observations(
    checkpoint_rows: list[dict[str, Any]],
    compile_rows: list[dict[str, Any]],
    attention_records: list[dict[str, Any]],
    correctness: dict[str, Any],
) -> list[dict[str, Any]]:
    observations: list[dict[str, Any]] = []

    def add(label: str, memory: Mapping[str, Any]) -> None:
        allocated = memory.get("peak_allocated_mib")
        reserved = memory.get("peak_reserved_mib")
        if _finite(allocated, minimum=0) and _finite(reserved, minimum=0):
            observations.append({"source": label, "peak_allocated_mib": float(allocated), "peak_reserved_mib": float(reserved)})

    for row in checkpoint_rows:
        add(f"checkpoint:{row.get('config_id')}", row)
    for row in compile_rows:
        add(f"compile:{row.get('config_id')}", row)
        add(
            f"compile-cold:{row.get('config_id')}",
            {
                "peak_allocated_mib": row.get("cold_peak_allocated_mib"),
                "peak_reserved_mib": row.get("cold_peak_reserved_mib"),
            },
        )
    for record in attention_records:
        memory = record.get("memory")
        if isinstance(memory, dict):
            add(f"attention:{record.get('case_id')}", memory)
    for case in correctness.get("cases", []):
        for implementation, result in case.get("implementations", {}).items():
            memory = result.get("memory") if isinstance(result, dict) else None
            if isinstance(memory, dict):
                add(f"correctness:{case.get('case_id')}:{implementation}", memory)
    return observations


def build_memory_evidence(
    *,
    mode: str,
    runtime: dict[str, Any],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    _require(observations or mode == "dry_run", "no CUDA peak-memory observations were recorded")
    peak_allocated = max((row["peak_allocated_mib"] for row in observations), default=0.0)
    peak_reserved = max((row["peak_reserved_mib"] for row in observations), default=0.0)
    source = max(observations, key=lambda row: row["peak_reserved_mib"])["source"] if observations else None
    allocator = runtime["allocator"]
    within = mode != "dry_run" and allocator.get("guard_applied") is True and peak_allocated <= peak_reserved + 1e-6 and peak_reserved <= ALLOCATOR_LIMIT_MIB + 1e-6
    return {
        "schema_version": SCHEMA_VERSION,
        "authoritative": mode == "formal_cuda",
        "evidence_class": _evidence_class(mode),
        "non_authoritative_reason": _non_authoritative_reason(mode, runtime.get("gpu", {}).get("name")),
        "allocator": {
            "allocator_fraction": allocator.get("allocator_fraction"),
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "guard_applied_before_first_cuda_allocation": allocator.get("applied_before_first_cuda_allocation", False),
        },
        "hard_limit_mib": HARD_LIMIT_MIB,
        "pytorch_peak_allocated_mib": round(peak_allocated, 6),
        "pytorch_peak_reserved_mib": round(peak_reserved, 6),
        "peak_reserved_source": source,
        "within_24gib": within,
        "formal_rtx4090_24gb_evidence": mode == "formal_cuda" and within,
        "process_observation_count": len(observations),
    }


def _csv_text(rows: Iterable[Mapping[str, Any]], fieldnames: Sequence[str]) -> str:
    buffer = StringIO(newline="")
    writer = csv.DictWriter(
        buffer,
        fieldnames=list(fieldnames),
        extrasaction="ignore",
        lineterminator="\n",
    )
    writer.writeheader()
    for row in rows:
        output: dict[str, Any] = {}
        for field in fieldnames:
            value = row.get(field)
            if isinstance(value, (dict, list, tuple)):
                value = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
            elif value is None:
                value = ""
            output[field] = value
        writer.writerow(output)
    return buffer.getvalue()


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    _write_text(path, json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n")


def _sanitized_unit_test_lines(output_path: Path, summary: Mapping[str, Any]) -> list[str]:
    """Extract only pinned test node IDs and terminal outcomes from pytest output."""
    _require(output_path.is_file(), "official unit-test output is missing")
    pattern = re.compile(
        r"^(tests/test_attention\.py::[A-Za-z0-9_]+(?:\[[A-Za-z0-9_.-]+\])?) "
        r"(PASSED|FAILED|SKIPPED|XFAIL|XPASS|ERROR)(?:\s+\(.*\))?(?:\s+\[[^\]]+\])?$"
    )
    lines: list[str] = []
    counts = {name: 0 for name in ("passed", "failed", "skipped", "xfailed", "xpassed", "errors")}
    status_keys = {
        "PASSED": "passed",
        "FAILED": "failed",
        "SKIPPED": "skipped",
        "XFAIL": "xfailed",
        "XPASS": "xpassed",
        "ERROR": "errors",
    }
    for raw_line in output_path.read_text(encoding="utf-8", errors="replace").splitlines():
        match = pattern.fullmatch(raw_line.strip())
        if match is None:
            continue
        sanitized = f"{match.group(1)} {match.group(2)}"
        _require(sanitized not in lines, "duplicate official unit-test outcome")
        lines.append(sanitized)
        counts[status_keys[match.group(2)]] += 1
    _require(len(lines) == summary["total"], "official unit-test output row count mismatch")
    _require(
        {line.rsplit(" ", 1)[0] for line in lines} == set(OFFICIAL_TEST_NODE_IDS),
        "official unit-test node IDs do not match the pinned suite",
    )
    for key, value in counts.items():
        _require(value == summary[key], f"official unit-test output {key} count mismatch")
    return lines


def _unit_test_text(
    payload: dict[str, Any],
    *,
    mode: str,
    gpu_name: str | None,
    case_lines: Sequence[str],
) -> str:
    summary = payload["summary"]
    lines = [
        "A2-K official attention tests (sanitized output)",
        "Command: uv run pytest tests/test_attention.py -v",
        f"Evidence class: {_evidence_class(mode)}",
        f"Authoritative RTX 4090 24GB evidence: {'yes' if mode == 'formal_cuda' else 'no'}",
        f"GPU: {gpu_name or 'none (CPU dry-run)'}",
        f"Exit code: {payload.get('return_code')}",
        f"Collected: {summary['total']}",
        f"Passed: {summary['passed']}",
        f"Failed: {summary['failed']}",
        f"Skipped: {summary['skipped']}",
        f"Xfailed: {summary['xfailed']}",
        f"Xpassed: {summary['xpassed']}",
        f"Errors: {summary['errors']}",
    ]
    reason = _non_authoritative_reason(mode, gpu_name)
    if reason:
        lines.append(f"Notice: {reason}")
    if case_lines:
        lines.extend(["", "Test outcomes:", *case_lines])
    lines.append("Raw terminal output is retained privately and is not part of this public artifact.")
    return "\n".join(lines) + "\n"


def _plot_checkpoint(path: Path, rows: list[dict[str, Any]], *, mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    context = 8 if mode == "dry_run" else 1024
    selected = sorted((row for row in rows if row.get("context_length") == context), key=lambda row: row["checkpoint_block_size"])
    labels = ["none" if row["checkpoint_block_size"] == 0 else str(row["checkpoint_block_size"]) for row in selected]
    allocated = [float(row.get("peak_allocated_mib") or 0) for row in selected]
    latency = [float(row.get("step_time_ms_p50") or 0) for row in selected]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    bars = axis.bar(labels, allocated, color="#4C78A8", label="Peak allocated")
    axis.set_xlabel("Checkpoint block size (none = no checkpoint)")
    axis.set_ylabel("Peak allocated (MiB)")
    axis.set_title("Activation checkpointing memory/time trade-off")
    time_axis = axis.twinx()
    time_axis.plot(labels, latency, marker="o", color="#E45756", linewidth=2, label="Step p50")
    time_axis.set_ylabel("Training-step p50 (ms)")
    axis.legend(handles=[bars, time_axis.lines[0]], labels=["Peak allocated", "Step p50"], loc="best")
    figure.savefig(path, dpi=120, metadata={"Software": "A2-K summarize"})
    plt.close(figure)


def _plot_attention(path: Path, rows: list[dict[str, Any]], *, mode: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    candidates = [
        row
        for row in rows
        if row.get("phase") == "forward"
        and row.get("head_dim") == (32 if mode == "dry_run" else 128)
        and row.get("p50_ms") not in (None, "")
        and row.get("status") in SUCCESS_STATUSES
    ]
    figure, axis = plt.subplots(figsize=(7.2, 4.2), constrained_layout=True)
    for implementation in ("eager", "compiled", "triton"):
        selected = sorted((row for row in candidates if row.get("implementation") == implementation), key=lambda row: row["sequence_length"])
        if selected:
            axis.plot(
                [row["sequence_length"] for row in selected],
                [row["p50_ms"] for row in selected],
                marker="o",
                linewidth=2,
                label=implementation,
            )
    axis.set_xlabel("Sequence length")
    axis.set_ylabel("Forward p50 latency (ms)")
    axis.set_title("Causal attention latency at fixed head dimension")
    axis.set_yscale("log")
    if mode != "dry_run":
        axis.set_xscale("log", base=2)
    axis.grid(True, which="both", alpha=0.25)
    axis.legend(loc="best")
    figure.savefig(path, dpi=120, metadata={"Software": "A2-K summarize"})
    plt.close(figure)


def _publish(staging_results: Path, staging_assets: Path, results_dir: Path, assets_dir: Path) -> None:
    expected_results = {path.name for path in staging_results.iterdir()}
    expected_assets = {path.name for path in staging_assets.iterdir()}
    _require(not results_dir.is_symlink() and not assets_dir.is_symlink(), "public destinations must not be symlinks")
    _require(not results_dir.exists() or results_dir.is_dir(), "results destination is not a directory")
    _require(not assets_dir.exists() or assets_dir.is_dir(), "assets destination is not a directory")

    backup_root = staging_results.parent
    previous_results = backup_root / "previous-results"
    previous_assets = backup_root / "previous-assets"
    had_results = results_dir.exists()
    had_assets = assets_dir.exists()
    published_results = False
    published_assets = False
    try:
        # Move complete old trees aside first, then install the staged trees.
        # The temporary-directory cleanup removes the old trees only after the
        # exact final structure and attachment budget have been revalidated.
        if had_results:
            os.replace(results_dir, previous_results)
        if had_assets:
            os.replace(assets_dir, previous_assets)
        os.replace(staging_results, results_dir)
        published_results = True
        os.replace(staging_assets, assets_dir)
        published_assets = True
        _require({path.name for path in results_dir.iterdir()} == expected_results, "published results tree is not exact")
        _require({path.name for path in assets_dir.iterdir()} == expected_assets, "published assets tree is not exact")
        total_bytes = sum(path.stat().st_size for path in results_dir.iterdir()) + sum(path.stat().st_size for path in assets_dir.iterdir())
        _require(total_bytes <= 2 * 1024 * 1024, "published results and assets exceed the 2 MiB limit")
    except BaseException:
        if published_assets and assets_dir.exists():
            os.replace(assets_dir, staging_assets)
        if published_results and results_dir.exists():
            os.replace(results_dir, staging_results)
        if had_assets and previous_assets.exists():
            os.replace(previous_assets, assets_dir)
        if had_results and previous_results.exists():
            os.replace(previous_results, results_dir)
        raise


def summarize(raw_dir: Path, results_dir: Path, assets_dir: Path) -> dict[str, Any]:
    raw_dir = _within_workspace(raw_dir, "raw directory", allow_runtime=True)
    results_dir = _within_workspace(results_dir, "results directory")
    assets_dir = _within_workspace(assets_dir, "assets directory")
    manifest = _validate_manifest(raw_dir)
    mode = manifest["execution_mode"]
    unit_tests = _validate_unit_tests(_load_json(raw_dir / "unit_tests" / "result.json"), mode=mode)
    unit_test_runtime = _validate_runtime(
        _load_json(raw_dir / "unit_tests" / "runtime_guard.json"),
        mode=mode,
        label="official unit tests",
    )
    correctness, correctness_runtimes = _validate_correctness(_load_json(raw_dir / "correctness" / "result.json"), mode=mode)
    checkpoint_rows, checkpoint_runtimes = _validate_checkpointing(
        _load_json(raw_dir / "checkpointing" / "result.json"),
        mode=mode,
        expected_seed=manifest["seed"],
    )
    compile_rows, compile_runtimes = _validate_compile(
        _load_json(raw_dir / "compile_comparison" / "result.json"),
        mode=mode,
        expected_seed=manifest["seed"],
    )
    attention_records, attention_runtimes = _load_attention(raw_dir, mode=mode, expected_seed=manifest["seed"])
    runtimes = [unit_test_runtime] + correctness_runtimes + checkpoint_runtimes + compile_runtimes + attention_runtimes
    representative = _consistent_runtime(runtimes, mode=mode)
    gpu_name = representative.get("gpu", {}).get("name")
    evidence = _evidence_class(mode)
    authoritative = mode == "formal_cuda"

    checkpoint_public = [
        {
            **{field: row.get(field) for field in CHECKPOINT_COLUMNS if field not in {"authoritative", "evidence_class"}},
            "authoritative": authoritative,
            "evidence_class": evidence,
        }
        for row in checkpoint_rows
    ]
    compile_public: list[dict[str, Any]] = []
    for row in compile_rows:
        public_row = {field: row.get(field) for field in COMPILE_COLUMNS if field not in {"authoritative", "evidence_class", "raw_samples_retained_privately"}}
        # Event-duration runs can contain tens of thousands of samples.  Their
        # exact arrays remain private; count and p20/p50/p80 stay public.
        samples_private = row.get("workload") == "attention" and mode != "dry_run"
        if samples_private:
            public_row["steady_ms_samples"] = []
        public_row["raw_samples_retained_privately"] = samples_private
        public_row["authoritative"] = authoritative
        public_row["evidence_class"] = evidence
        compile_public.append(public_row)
    attention_public = _attention_public_rows(attention_records, mode=mode)
    baseline_public = [row for row in attention_public if row["implementation"] == "eager"]
    correctness_public = copy.deepcopy(correctness)
    correctness_public.pop("case_runtime_evidence", None)
    correctness_public["authoritative"] = authoritative
    correctness_public["evidence_class"] = evidence
    correctness_public["non_authoritative_reason"] = _non_authoritative_reason(mode, gpu_name)

    observations = _memory_observations(checkpoint_rows, compile_rows, attention_records, correctness)
    memory_evidence = build_memory_evidence(mode=mode, runtime=representative, observations=observations)
    gpu_public = copy.deepcopy(representative.get("gpu")) if mode != "dry_run" else None
    run_metadata = {
        "schema_version": SCHEMA_VERSION,
        "authoritative": authoritative,
        "evidence_class": evidence,
        "non_authoritative_reason": _non_authoritative_reason(mode, gpu_name),
        "commit": STARTER_COMMIT,
        "seed": manifest["seed"],
        "command": (
            "python -m student_scripts.a2k.run_suite --raw-dir .runtime/a2k/raw/run"
            + (" --development-cuda" if mode == "development_cuda" else " --dry-run" if mode == "dry_run" else "")
        ),
        "execution": {
            "mode": mode,
            "single_visible_gpu": mode != "dry_run",
            "serial": True,
            "fresh_process_per_attention_case": True,
            "fresh_process_per_checkpoint_case": True,
            "fresh_process_per_compile_case": True,
            "concurrent_benchmarks": False,
        },
        "gpu": gpu_public,
        "software": copy.deepcopy(representative.get("software")),
        "allocator": {
            "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
            "allocator_fraction": representative["allocator"].get("allocator_fraction"),
            "applied_before_first_cuda_allocation": representative["allocator"].get("applied_before_first_cuda_allocation", False),
        },
        "tf32": {
            "performance_policy": representative.get("tf32_policy"),
            "fp32_correctness_policy": "ieee",
        },
        "attention_timer": {
            "name": "triton.testing.do_bench" if mode != "dry_run" else "time.perf_counter",
            "warmup_ms": 100 if mode != "dry_run" else None,
            "rep_ms": 300 if mode != "dry_run" else None,
            "quantiles": [0.2, 0.5, 0.8],
            "synchronized_boundaries": mode != "dry_run",
        },
        "checkpoint_measurement": {
            "warmup_steps": 1 if mode == "dry_run" else 3,
            "measurement_steps": 1 if mode == "dry_run" else 5,
            "parameters": "fp32",
            "autocast": "disabled" if mode == "dry_run" else "bf16",
            "optimizer": "AdamW",
        },
        "compile": {
            "backend": "eager" if mode == "dry_run" else "inductor",
            "dynamic": False,
            "cold_start_separate_from_steady_state": True,
            "private_cache_per_case": True,
        },
        "matrix": {
            "attention_case_count": len(attention_public),
            "checkpoint_case_count": len(checkpoint_public),
            "compile_case_count": len(compile_public),
            "correctness_case_count": len(correctness.get("cases", [])),
            "no_silent_shape_reduction": mode != "dry_run",
        },
    }

    public_values = [
        checkpoint_public,
        compile_public,
        attention_public,
        correctness_public,
        memory_evidence,
        run_metadata,
    ]
    for value in public_values:
        _assert_public(value)

    _require(results_dir.parent == assets_dir.parent, "results and assets directories must be siblings")
    publication_root = results_dir.parent
    publication_root.mkdir(parents=True, exist_ok=True)
    # Stage on the destination filesystem so atomic os.replace remains valid
    # even when the workspace and system temporary directory are different
    # mounts.
    with tempfile.TemporaryDirectory(prefix=".a2k-summarize-", dir=publication_root) as temporary:
        temporary_root = Path(temporary)
        staging_results = temporary_root / "results"
        staging_assets = temporary_root / "assets"
        staging_results.mkdir()
        staging_assets.mkdir()
        _write_json(staging_results / "correctness.json", correctness_public)
        unit_test_lines = _sanitized_unit_test_lines(raw_dir / "unit_tests" / "output.txt", unit_tests["summary"])
        _write_text(
            staging_results / "unit_tests.txt",
            _unit_test_text(unit_tests, mode=mode, gpu_name=gpu_name, case_lines=unit_test_lines),
        )
        _write_text(staging_results / "checkpointing.csv", _csv_text(checkpoint_public, CHECKPOINT_COLUMNS))
        _write_text(staging_results / "attention_baseline.csv", _csv_text(baseline_public, ATTENTION_COLUMNS))
        _write_text(staging_results / "compile_comparison.csv", _csv_text(compile_public, COMPILE_COLUMNS))
        _write_text(staging_results / "flash_benchmark.csv", _csv_text(attention_public, ATTENTION_COLUMNS))
        _write_json(staging_results / "memory_evidence.json", memory_evidence)
        _write_json(staging_results / "run_metadata.json", run_metadata)

        matplotlib_cache = PROJECT_ROOT / ".runtime" / "a2k" / "matplotlib"
        matplotlib_cache.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(matplotlib_cache))
        _plot_checkpoint(staging_assets / "checkpoint_tradeoff.png", checkpoint_rows, mode=mode)
        _plot_attention(staging_assets / "attention_latency.png", attention_public, mode=mode)

        expected_results = {
            "correctness.json",
            "unit_tests.txt",
            "checkpointing.csv",
            "attention_baseline.csv",
            "compile_comparison.csv",
            "flash_benchmark.csv",
            "memory_evidence.json",
            "run_metadata.json",
        }
        _require({path.name for path in staging_results.iterdir()} == expected_results, "public result staging set is incomplete")
        _require(len(list(staging_assets.glob("*.png"))) >= 2, "at least two PNG assets are required")
        attachment_bytes = sum(path.stat().st_size for path in staging_results.iterdir()) + sum(path.stat().st_size for path in staging_assets.iterdir())
        _require(attachment_bytes <= 2 * 1024 * 1024, "public results and assets exceed the 2 MiB limit")
        for path in [*staging_results.iterdir(), *staging_assets.iterdir()]:
            _require(path.stat().st_size <= 5 * 1024 * 1024, f"public artifact exceeds 5 MiB: {path.name}")
        _publish(staging_results, staging_assets, results_dir, assets_dir)

    return {
        "status": "published",
        "authoritative": authoritative,
        "evidence_class": evidence,
        "result_file_count": 8,
        "asset_file_count": 2,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Fail-closed A2-K raw-evidence validator and public summarizer.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--results-dir", type=Path, required=True)
    parser.add_argument("--assets-dir", type=Path, required=True)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        result = summarize(args.raw_dir, args.results_dir, args.assets_dir)
    except EvidenceValidationError as exc:
        print(f"validation failed: {exc}", file=os.sys.stderr)
        return 2
    print(f"public result files: {result['result_file_count']}")
    print(f"public PNG assets: {result['asset_file_count']}")
    print(f"evidence class: {result['evidence_class']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
