"""Collect Task 4 peak-memory data and safely replay recorded CUDA OOMs."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Literal

from profiling.collect_utils import command_display, failure_kind, publish_files_transactionally, require_cuda
from profiling.summarize import read_jsonl, write_memory_csv


ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results" / "memory"
LOCAL_SNAPSHOTS = ROOT / "local_artifacts" / "memory"
MODES = ("forward", "train_step")
DTYPES = ("fp32", "bf16")
MEMORY_FILES = ("runs.jsonl", "failures.jsonl", "peaks.csv", "run_metadata.json")
RunStatus = Literal["success", "cuda_oom", "subprocess_failed"]
RunIdentity = tuple[str, int, int, str, str]
_MEMORY_STAT_FIELDS = (
    "active_bytes",
    "peak_active_bytes",
    "allocated_bytes",
    "peak_allocated_bytes",
    "reserved_bytes",
    "peak_reserved_bytes",
)
_MEMORY_VALUE_FIELDS = (*_MEMORY_STAT_FIELDS, "free_bytes", "total_bytes", "requested_allocation_bytes")
_FAILURE_SCOPES = {"initialization", "warmup", "measurement"}
_FAILURE_PHASES = {"forward", "backward", "optimizer"}
_PEAK_SCOPES = {"initialization", "warmup", "post_warmup_measurement"}


def run_name(*, model_size: str, context_length: int, batch_size: int, mode: str, dtype: str) -> str:
    return f"{model_size}_ctx{context_length}_bs{batch_size}_{mode}_{dtype}"


def run_identity(*, model_size: str, context_length: int, batch_size: int, mode: str, dtype: str) -> RunIdentity:
    return (model_size, context_length, batch_size, mode, dtype)


def identity_from_failure(record: dict[str, Any]) -> RunIdentity | None:
    try:
        model_size = record["model_size"]
        context_length = record["context_length"]
        batch_size = record["batch_size"]
        mode = record["mode"]
        dtype = record["dtype"]
    except KeyError:
        return None
    if not isinstance(model_size, str) or not isinstance(context_length, int) or not isinstance(batch_size, int):
        return None
    if not isinstance(mode, str) or not isinstance(dtype, str):
        return None
    return run_identity(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
    )


def identity_from_success(record: dict[str, Any]) -> RunIdentity | None:
    model = record.get("model_config")
    run = record.get("run_config")
    if not isinstance(model, dict) or not isinstance(run, dict):
        return None
    try:
        return run_identity(
            model_size=str(run["model_size"]),
            context_length=int(model["context_length"]),
            batch_size=int(model["batch_size"]),
            mode=str(run["mode"]),
            dtype=str(run["precision"]),
        )
    except (KeyError, TypeError, ValueError):
        return None


def command_for(
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    output: Path,
    snapshots: Path,
    failure_output: Path | None = None,
) -> list[str]:
    name = run_name(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
    )
    command = [
        sys.executable,
        "profiling/benchmark.py",
        "--model-size",
        model_size,
        "--context-length",
        str(context_length),
        "--batch-size",
        str(batch_size),
        "--mode",
        mode,
        "--dtype",
        dtype,
        "--seed",
        "0",
        "--warmup",
        "5",
        "--steps",
        "1",
        "--track-memory",
        "--memory-snapshot",
        str(snapshots / f"{name}.pickle"),
        "--output",
        str(output),
    ]
    if failure_output is not None:
        command.extend(("--failure-output", str(failure_output)))
    return command


def _unavailable_telemetry() -> dict[str, Any]:
    return {
        "schema_version": None,
        "failure_scope": "unavailable",
        "failure_phase": None,
        "peak_scope": "unavailable",
        "memory": {
            "telemetry_status": "unavailable",
            "unavailable_fields": list(_MEMORY_VALUE_FIELDS),
            "statistics_bytes": {field: None for field in _MEMORY_STAT_FIELDS},
            "free_bytes": None,
            "total_bytes": None,
            "requested_allocation_bytes": None,
        },
        "environment": {
            "device_name": None,
            "torch_version": None,
            "cuda_version": None,
            "python_version": None,
        },
    }


def _nonnegative_int_or_none(value: Any) -> int | None:
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _sanitize_failure_telemetry(raw: Any, *, expected: RunIdentity) -> dict[str, Any] | None:
    """Accept only the fixed, path-free schema emitted by benchmark.py."""

    if not isinstance(raw, dict) or raw.get("exception") != "cuda_oom":
        return None
    model = raw.get("model_config")
    run = raw.get("run_config")
    if not isinstance(model, dict) or not isinstance(run, dict):
        return None
    observed = run_identity(
        model_size=str(run.get("model_size", "")),
        context_length=_nonnegative_int_or_none(model.get("context_length")) or -1,
        batch_size=_nonnegative_int_or_none(model.get("batch_size")) or -1,
        mode=str(run.get("mode", "")),
        dtype=str(run.get("precision", "")),
    )
    if observed != expected:
        return None
    scope = raw.get("failure_scope")
    phase = raw.get("failure_phase")
    peak_scope = raw.get("peak_scope")
    if scope not in _FAILURE_SCOPES or phase not in _FAILURE_PHASES | {None} or peak_scope not in _PEAK_SCOPES:
        return None
    raw_memory = raw.get("memory")
    raw_stats = raw_memory.get("statistics_bytes") if isinstance(raw_memory, dict) else None
    if not isinstance(raw_stats, dict):
        raw_stats = {}
    values = {field: _nonnegative_int_or_none(raw_stats.get(field)) for field in _MEMORY_STAT_FIELDS}
    values.update(
        {
            field: _nonnegative_int_or_none(raw_memory.get(field)) if isinstance(raw_memory, dict) else None
            for field in ("free_bytes", "total_bytes", "requested_allocation_bytes")
        }
    )
    available = [field for field, value in values.items() if value is not None]
    raw_environment = raw.get("environment")
    if not isinstance(raw_environment, dict):
        raw_environment = {}
    environment = {
        field: raw_environment.get(field) if isinstance(raw_environment.get(field), str) else None
        for field in ("device_name", "torch_version", "cuda_version", "python_version")
    }
    return {
        "schema_version": raw.get("schema_version") if isinstance(raw.get("schema_version"), int) else None,
        "failure_scope": scope,
        "failure_phase": phase,
        "peak_scope": peak_scope,
        "memory": {
            "telemetry_status": "available" if len(available) == len(values) else "partial" if available else "unavailable",
            "unavailable_fields": [field for field, value in values.items() if value is None],
            "statistics_bytes": {field: values[field] for field in _MEMORY_STAT_FIELDS},
            "free_bytes": values["free_bytes"],
            "total_bytes": values["total_bytes"],
            "requested_allocation_bytes": values["requested_allocation_bytes"],
        },
        "environment": environment,
    }


def read_failure_telemetry(path: Path | None, *, expected: RunIdentity) -> dict[str, Any]:
    if path is None or not path.is_file():
        return _unavailable_telemetry()
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return _unavailable_telemetry()
    return _sanitize_failure_telemetry(raw, expected=expected) or _unavailable_telemetry()


def failure_record(
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    completed: subprocess.CompletedProcess[str],
    failure_output: Path | None = None,
) -> tuple[RunStatus, dict[str, Any]]:
    expected = run_identity(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
    )
    telemetry = read_failure_telemetry(failure_output, expected=expected)
    classified = failure_kind(completed)
    status: RunStatus = "cuda_oom" if telemetry["failure_scope"] != "unavailable" or classified == "cuda_oom" else "subprocess_failed"
    record: dict[str, Any] = {
        "schema_version": 2,
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "model_size": model_size,
        "context_length": context_length,
        "batch_size": batch_size,
        "mode": mode,
        "dtype": dtype,
        "stage": "benchmark subprocess",
        "exception": status,
        "return_code": completed.returncode,
    }
    if status == "cuda_oom":
        record.update({name: value for name, value in telemetry.items() if name != "schema_version"})
        record["telemetry_schema_version"] = telemetry["schema_version"]
        record["oom_telemetry_available"] = telemetry["memory"]["telemetry_status"] != "unavailable"
    return status, record


def append_failure(
    path: Path,
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    completed: subprocess.CompletedProcess[str],
    failure_output: Path | None = None,
) -> RunStatus:
    status, record = failure_record(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
        completed=completed,
        failure_output=failure_output,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as output_file:
        output_file.write(json.dumps(record, sort_keys=True) + "\n")
    return status


def telemetry_path(directory: Path, *, model_size: str, context_length: int, batch_size: int, mode: str, dtype: str) -> Path:
    name = run_name(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
    )
    return directory / ".failure_telemetry" / f"{name}.json"


def run_one(
    *,
    model_size: str,
    context_length: int,
    batch_size: int,
    mode: str,
    dtype: str,
    output: Path,
    snapshots: Path,
    failures: Path | None,
    failure_records: list[dict[str, Any]] | None = None,
    telemetry_dir: Path | None = None,
) -> RunStatus:
    failure_output = telemetry_path(
        telemetry_dir or LOCAL_SNAPSHOTS,
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
    )
    # Avoid attributing a later non-OOM subprocess failure to stale telemetry.
    failure_output.unlink(missing_ok=True)
    command = command_for(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failure_output=failure_output,
    )
    print("Running:", command_display(command), flush=True)
    completed = subprocess.run(command, cwd=ROOT, check=False, text=True, stderr=subprocess.PIPE)
    if completed.returncode == 0:
        return "success"
    if completed.stderr:
        print(completed.stderr, file=sys.stderr, end="")
    status, record = failure_record(
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        mode=mode,
        dtype=dtype,
        completed=completed,
        failure_output=failure_output,
    )
    if failures is not None:
        failures.parent.mkdir(parents=True, exist_ok=True)
        with failures.open("a", encoding="utf-8") as output_file:
            output_file.write(json.dumps(record, sort_keys=True) + "\n")
    if failure_records is not None:
        failure_records.append(record)
    return status


def run_with_fallback(*, context_length: int, mode: str, dtype: str, output: Path, snapshots: Path, failures: Path) -> None:
    status = run_one(
        model_size="xl",
        context_length=context_length,
        batch_size=4,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom" or context_length != 2048:
        return

    status = run_one(
        model_size="xl",
        context_length=2048,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom":
        return

    status = run_one(
        model_size="xl",
        context_length=1024,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )
    if status != "cuda_oom":
        return

    run_one(
        model_size="large",
        context_length=2048,
        batch_size=1,
        mode=mode,
        dtype=dtype,
        output=output,
        snapshots=snapshots,
        failures=failures,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Collect the Task 4 XL memory matrix and explicit OOM fallbacks.")
    parser.add_argument("--output-dir", type=Path, default=RESULTS)
    parser.add_argument("--snapshot-dir", type=Path, default=LOCAL_SNAPSHOTS)
    parser.add_argument("--retry-oom", action="store_true", help="Replay only existing cuda_oom records; preserve every other result.")
    parser.add_argument("--dry-run", action="store_true", help="Print planned commands without requiring CUDA or writing files.")
    return parser


def compact_metadata(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Keep reproducibility data but omit raw timings and internal filesystem paths."""

    metadata: list[dict[str, Any]] = []
    for record in records:
        model = record["model_config"]
        run = record["run_config"]
        memory = record.get("memory") or {}
        metadata.append(
            {
                "timestamp_utc": record["timestamp_utc"],
                "model_size": run["model_size"],
                "mode": run["mode"],
                "dtype": run["precision"],
                "batch_size": model["batch_size"],
                "context_length": model["context_length"],
                "warmup_steps": run["warmup_steps"],
                "measurement_steps": run["measurement_steps"],
                "snapshot_file": memory.get("snapshot_file"),
                "environment": record["environment"],
                "command": record["command"],
            }
        )
    return metadata


def _write_jsonl_atomically(path: Path, records: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text("".join(json.dumps(record, sort_keys=True) + "\n" for record in records), encoding="utf-8")
    temporary.replace(path)


def _write_json_atomically(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    temporary.replace(path)


def publish_memory_artifacts(*, output_dir: Path, records: list[dict[str, Any]], failures: list[dict[str, Any]]) -> None:
    """Stage and transactionally publish the four derived memory artifacts."""

    output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".memory-publish-", dir=output_dir) as temporary_directory:
        staged = Path(temporary_directory)
        _write_jsonl_atomically(staged / "runs.jsonl", records)
        _write_jsonl_atomically(staged / "failures.jsonl", failures)
        write_memory_csv(records, staged / "peaks.csv")
        _write_json_atomically(staged / "run_metadata.json", compact_metadata(records))
        publish_files_transactionally(tuple((staged / name, output_dir / name) for name in MEMORY_FILES))


def _commands_for_oom_records(*, failures: list[dict[str, Any]], output_dir: Path, snapshots: Path) -> list[tuple[RunIdentity, list[str]]]:
    seen: set[RunIdentity] = set()
    commands: list[tuple[RunIdentity, list[str]]] = []
    for record in failures:
        if record.get("exception") != "cuda_oom":
            continue
        identity = identity_from_failure(record)
        if identity is None or identity in seen:
            continue
        seen.add(identity)
        model_size, context_length, batch_size, mode, dtype = identity
        name = run_name(
            model_size=model_size,
            context_length=context_length,
            batch_size=batch_size,
            mode=mode,
            dtype=dtype,
        )
        commands.append(
            (
                identity,
                command_for(
                    model_size=model_size,
                    context_length=context_length,
                    batch_size=batch_size,
                    mode=mode,
                    dtype=dtype,
                    output=output_dir / ".retry_oom" / f"{name}.jsonl",
                    snapshots=snapshots,
                    failure_output=output_dir / ".retry_oom" / ".failure_telemetry" / f"{name}.json",
                ),
            )
        )
    return commands


def retry_existing_oom(*, output_dir: Path, snapshots: Path, dry_run: bool = False) -> None:
    """Replay exactly the historic CUDA OOM rows, merging any new outcome safely."""

    failures_path = output_dir / "failures.jsonl"
    failures = read_jsonl(failures_path) if failures_path.exists() else []
    commands = _commands_for_oom_records(failures=failures, output_dir=output_dir, snapshots=snapshots)
    if dry_run:
        for _, command in commands:
            print("Planned OOM retry:", command_display(command), flush=True)
        if not commands:
            print("No existing cuda_oom records to replay.", flush=True)
        return
    if not commands:
        print("No existing cuda_oom records to replay; no result files were changed.", flush=True)
        return
    require_cuda()
    records_path = output_dir / "runs.jsonl"
    existing_records = read_jsonl(records_path) if records_path.exists() else []
    target_identities = {identity for identity, _ in commands}
    merged_records = list(existing_records)
    replacement_failures: list[dict[str, Any]] = []
    successful_retries = 0
    with tempfile.TemporaryDirectory(prefix=".retry-oom-", dir=output_dir) as temporary_directory:
        staging = Path(temporary_directory)
        for identity, _ in commands:
            model_size, context_length, batch_size, mode, dtype = identity
            name = run_name(
                model_size=model_size,
                context_length=context_length,
                batch_size=batch_size,
                mode=mode,
                dtype=dtype,
            )
            attempt_output = staging / f"{name}.jsonl"
            attempt_failures: list[dict[str, Any]] = []
            status = run_one(
                model_size=model_size,
                context_length=context_length,
                batch_size=batch_size,
                mode=mode,
                dtype=dtype,
                output=attempt_output,
                snapshots=snapshots,
                failures=None,
                failure_records=attempt_failures,
                telemetry_dir=staging,
            )
            if status == "success":
                produced = read_jsonl(attempt_output)
                if len(produced) != 1 or identity_from_success(produced[0]) != identity:
                    raise RuntimeError(f"OOM retry for {name} succeeded without exactly one matching result record; no artifacts were replaced.")
                merged_records = [record for record in merged_records if identity_from_success(record) != identity]
                merged_records.append(produced[0])
                successful_retries += 1
                continue
            if status != "cuda_oom":
                raise RuntimeError(
                    f"OOM retry for {name} ended as {status}, not CUDA OOM; "
                    "the historical OOM record was retained."
                )
            if len(attempt_failures) != 1:
                raise RuntimeError(f"OOM retry for {name} failed without a structured replacement record; no artifacts were replaced.")
            if attempt_failures[0].get("exception") != "cuda_oom":
                raise RuntimeError(f"OOM retry for {name} did not produce a CUDA-OOM replacement record; no artifacts were replaced.")
            replacement_failures.append(attempt_failures[0])

    retained_failures = [
        record
        for record in failures
        if not (record.get("exception") == "cuda_oom" and identity_from_failure(record) in target_identities)
    ]
    publish_memory_artifacts(
        output_dir=output_dir,
        records=merged_records,
        failures=[*retained_failures, *replacement_failures],
    )
    print(
        f"Replayed {len(commands)} CUDA OOM record(s): {successful_retries} succeeded and were merged; "
        f"{len(replacement_failures)} still failed and were replaced with fresh telemetry.",
        flush=True,
    )


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.retry_oom:
        retry_existing_oom(output_dir=args.output_dir, snapshots=args.snapshot_dir, dry_run=args.dry_run)
        return
    output = args.output_dir / "runs.jsonl"
    initial_commands = [
        command_for(
            model_size="xl",
            context_length=context_length,
            batch_size=4,
            mode=mode,
            dtype=dtype,
            output=output,
            snapshots=args.snapshot_dir,
        )
        for dtype in DTYPES
        for context_length in (128, 2048)
        for mode in MODES
    ]
    if args.dry_run:
        for command in initial_commands:
            print("Planned:", command_display(command), flush=True)
        print("If an XL/context-2048 run OOMs, the collector retries batch 1, then XL/context-1024 batch 1, then Large/context-2048 batch 1.")
        return
    try:
        require_cuda()
    except RuntimeError as error:
        parser.error(str(error))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.write_text("", encoding="utf-8")
    failures_path = args.output_dir / "failures.jsonl"
    failures_path.write_text("", encoding="utf-8")
    for dtype in DTYPES:
        for context_length in (128, 2048):
            for mode in MODES:
                run_with_fallback(
                    context_length=context_length,
                    mode=mode,
                    dtype=dtype,
                    output=output,
                    snapshots=args.snapshot_dir,
                    failures=failures_path,
                )
    publish_memory_artifacts(
        output_dir=args.output_dir,
        records=read_jsonl(output),
        failures=read_jsonl(failures_path),
    )


if __name__ == "__main__":
    main()
