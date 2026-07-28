"""Build measurement-only, submission-safe summaries from torch Chrome traces.

``torch.profiler.key_averages()`` aggregates every event in the profiler
context.  The Task 2 protocol deliberately keeps one warm-up step inside that
context, so its output is not an appropriate final result table.  This module
uses the Chrome trace itself as the source of truth and only reduces work
contained in the explicit ``profile/measure`` user annotation.

The module deliberately depends on the Python standard library only.  It can
therefore repair the Task 2 CSV and metadata on a machine that has the saved
traces but no CUDA runtime.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import tempfile
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRACE_DIR = ROOT / "local_artifacts" / "profile"
DEFAULT_RUNS = ROOT / "results" / "profile" / "runs.jsonl"
DEFAULT_OUTPUT = ROOT / "results" / "profile" / "trace_summary.csv"
DEFAULT_METADATA_OUTPUT = ROOT / "results" / "profile" / "run_metadata.json"

MEASURE_RANGE = "profile/measure"
WARMUP_RANGE = "profile/warmup"
TOP_PHASES = ("forward", "backward", "optimizer")
ATTENTION_RANGES = ("attention/scores", "attention/softmax", "attention/value")
PHYSICAL_GPU_CATEGORIES = frozenset({"kernel", "gpu_memcpy", "gpu_memset"})

TASK2_RUN_NAMES = tuple(
    f"{model_size}_ctx{context_length}_train_step_fp32"
    for model_size in ("small", "medium")
    for context_length in (256, 512, 1024)
)

CSV_FIELDNAMES = [
    "run_name",
    "row_type",
    "name",
    "stage",
    "activity_type",
    "calls",
    "range_duration_us",
    "cpu_time_total_us",
    "cuda_time_total_us",
    "kernel_calls",
    "inclusive",
    "notes",
]


class TraceSummaryError(ValueError):
    """Raised when a trace cannot support a measurement-only summary."""


@dataclass(frozen=True)
class TimeRange:
    name: str
    start_us: float
    duration_us: float

    @property
    def end_us(self) -> float:
        return self.start_us + self.duration_us

    def contains(self, timestamp_us: float) -> bool:
        return self.start_us <= timestamp_us < self.end_us


@dataclass(frozen=True)
class ArtifactReport:
    """Validation facts consumed by the repair entrypoint and CLI."""

    run_names: tuple[str, ...]
    row_count: int
    metadata_count: int


def _number(event: Mapping[str, Any], key: str, *, trace_path: Path) -> float:
    value = event.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise TraceSummaryError(f"{trace_path}: trace event has no finite {key!r}: {event.get('name')!r}.")
    return float(value)


def _complete_event(event: Mapping[str, Any], *, trace_path: Path) -> bool:
    """Return whether ``event`` is a duration-bearing Chrome trace slice."""

    if event.get("ph") != "X":
        return False
    _number(event, "ts", trace_path=trace_path)
    duration = _number(event, "dur", trace_path=trace_path)
    if duration < 0:
        raise TraceSummaryError(f"{trace_path}: negative duration for {event.get('name')!r}.")
    return True


def _event_range(event: Mapping[str, Any], *, trace_path: Path) -> TimeRange:
    name = event.get("name")
    if not isinstance(name, str) or not name:
        raise TraceSummaryError(f"{trace_path}: range event has no non-empty name.")
    return TimeRange(name=name, start_us=_number(event, "ts", trace_path=trace_path), duration_us=_number(event, "dur", trace_path=trace_path))


def _args(event: Mapping[str, Any]) -> Mapping[str, Any]:
    args = event.get("args")
    return args if isinstance(args, Mapping) else {}


def _external_id(event: Mapping[str, Any]) -> str | None:
    value = _args(event).get("External id")
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float, str)):
        return str(value)
    return None


def _load_events(trace_path: Path) -> list[dict[str, Any]]:
    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except FileNotFoundError as error:
        raise TraceSummaryError(f"Missing trace: {trace_path}") from error
    except json.JSONDecodeError as error:
        raise TraceSummaryError(f"Invalid JSON trace {trace_path}: {error.msg}.") from error
    if not isinstance(payload, Mapping):
        raise TraceSummaryError(f"{trace_path}: trace root must be a JSON object.")
    events = payload.get("traceEvents")
    if not isinstance(events, list):
        raise TraceSummaryError(f"{trace_path}: traceEvents must be a JSON array.")
    invalid = [event for event in events if not isinstance(event, dict)]
    if invalid:
        raise TraceSummaryError(f"{trace_path}: traceEvents contains a non-object event.")
    return events


def _user_ranges(events: Iterable[Mapping[str, Any]], *, trace_path: Path, name: str) -> list[TimeRange]:
    return [
        _event_range(event, trace_path=trace_path)
        for event in events
        if event.get("cat") == "user_annotation" and event.get("name") == name and _complete_event(event, trace_path=trace_path)
    ]


def _range_stage(timestamp_us: float, stage_ranges: Sequence[TimeRange], *, trace_path: Path, subject: str) -> str:
    matches = [time_range.name for time_range in stage_ranges if time_range.contains(timestamp_us)]
    if len(matches) != 1:
        description = "no" if not matches else "multiple"
        raise TraceSummaryError(f"{trace_path}: {subject} has {description} top-level phase association at {timestamp_us} us.")
    return matches[0]


def _require_inside(container: TimeRange, candidate: TimeRange, *, trace_path: Path) -> None:
    if candidate.start_us < container.start_us or candidate.end_us > container.end_us:
        raise TraceSummaryError(f"{trace_path}: {candidate.name!r} is not fully contained in {MEASURE_RANGE!r}.")


def _ranges_inside_measure(ranges: Iterable[TimeRange], measure: TimeRange, *, trace_path: Path) -> list[TimeRange]:
    """Select measurement ranges and reject a range that straddles its boundary.

    The trace intentionally contains same-named forward/backward/optimizer
    ranges inside ``profile/warmup``.  Ranges that are wholly outside the
    measurement window are therefore ignored rather than treated as errors.
    """

    selected: list[TimeRange] = []
    for time_range in ranges:
        overlaps_measure = time_range.start_us < measure.end_us and measure.start_us < time_range.end_us
        if not overlaps_measure:
            continue
        _require_inside(measure, time_range, trace_path=trace_path)
        selected.append(time_range)
    return selected


def _format_number(value: float | int | str | None) -> float | int | str:
    """Keep CSV cells blank when a metric does not apply to this row type."""

    return "" if value is None else value


def _row(**values: str | int | float | None) -> dict[str, str | int | float]:
    row: dict[str, str | int | float] = {field: "" for field in CSV_FIELDNAMES}
    for key, value in values.items():
        if key not in row:
            raise AssertionError(f"Unexpected trace summary column: {key}")
        row[key] = _format_number(value)
    return row


def summarize_trace(trace_path: Path, *, run_name: str, require_all_top_phases: bool = False) -> list[dict[str, str | int | float]]:
    """Reduce one Chrome trace into measurement-only range, CPU-op, and GPU rows.

    CUDA activity is never inferred from ``gpu_user_annotation``.  Each
    physical event is instead required to map through its External id to one
    unique CPU operation, then assigned to the containing top-level phase.
    """

    events = _load_events(trace_path)
    measure_ranges = _user_ranges(events, trace_path=trace_path, name=MEASURE_RANGE)
    if len(measure_ranges) != 1:
        raise TraceSummaryError(f"{trace_path}: expected exactly one {MEASURE_RANGE!r} user range, found {len(measure_ranges)}.")
    measure = measure_ranges[0]

    top_phase_ranges: list[TimeRange] = []
    for phase_name in TOP_PHASES:
        phase_ranges = _ranges_inside_measure(_user_ranges(events, trace_path=trace_path, name=phase_name), measure, trace_path=trace_path)
        if require_all_top_phases and not phase_ranges:
            raise TraceSummaryError(f"{trace_path}: no {phase_name!r} range inside {MEASURE_RANGE!r}.")
        top_phase_ranges.extend(phase_ranges)
    if not top_phase_ranges:
        raise TraceSummaryError(f"{trace_path}: no top-level phase range inside {MEASURE_RANGE!r}.")

    attention_ranges: list[TimeRange] = []
    for attention_name in ATTENTION_RANGES:
        for attention_range in _ranges_inside_measure(_user_ranges(events, trace_path=trace_path, name=attention_name), measure, trace_path=trace_path):
            parent_phase = _range_stage(
                attention_range.start_us,
                top_phase_ranges,
                trace_path=trace_path,
                subject=f"{attention_name!r} range",
            )
            if parent_phase != "forward":
                raise TraceSummaryError(f"{trace_path}: {attention_name!r} range is nested in {parent_phase!r}, not 'forward'.")
            attention_ranges.append(attention_range)

    cpu_ops: dict[str, Mapping[str, Any]] = {}
    selected_cpu_ops: list[tuple[Mapping[str, Any], str]] = []
    for event in events:
        if event.get("cat") != "cpu_op" or not _complete_event(event, trace_path=trace_path):
            continue
        external_id = _external_id(event)
        if external_id is not None:
            if external_id in cpu_ops:
                raise TraceSummaryError(f"{trace_path}: External id {external_id!r} maps to multiple cpu_op events.")
            cpu_ops[external_id] = event
        timestamp_us = _number(event, "ts", trace_path=trace_path)
        if measure.contains(timestamp_us):
            stage = _range_stage(timestamp_us, top_phase_ranges, trace_path=trace_path, subject=f"cpu_op {event.get('name')!r}")
            selected_cpu_ops.append((event, stage))

    if not selected_cpu_ops:
        raise TraceSummaryError(f"{trace_path}: no cpu_op events found inside {MEASURE_RANGE!r}.")

    range_groups: dict[tuple[str, str, str, str], list[float]] = defaultdict(list)
    range_groups[(MEASURE_RANGE, MEASURE_RANGE, "true", "measurement wall-clock CPU range; do not interpret as CUDA time")].append(measure.duration_us)
    for phase_range in top_phase_ranges:
        range_groups[(phase_range.name, phase_range.name, "true", "inclusive CPU annotation range")].append(phase_range.duration_us)
    for attention_range in attention_ranges:
        range_groups[(attention_range.name, "forward", "true", "inclusive nested range; do not add to forward")].append(attention_range.duration_us)

    cpu_groups: dict[tuple[str, str], list[float]] = defaultdict(list)
    for event, stage in selected_cpu_ops:
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TraceSummaryError(f"{trace_path}: cpu_op inside measurement has no name.")
        cpu_groups[(stage, name)].append(_number(event, "dur", trace_path=trace_path))

    cuda_groups: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    physical_event_count = 0
    for event in events:
        category = event.get("cat")
        if category not in PHYSICAL_GPU_CATEGORIES or not _complete_event(event, trace_path=trace_path):
            continue
        timestamp_us = _number(event, "ts", trace_path=trace_path)
        if not measure.contains(timestamp_us):
            continue
        physical_event_count += 1
        external_id = _external_id(event)
        if external_id is None:
            raise TraceSummaryError(f"{trace_path}: {category} activity inside measurement has no External id.")
        cpu_op = cpu_ops.get(external_id)
        if cpu_op is None:
            raise TraceSummaryError(f"{trace_path}: {category} activity External id {external_id!r} has no unique cpu_op association.")
        cpu_timestamp_us = _number(cpu_op, "ts", trace_path=trace_path)
        if not measure.contains(cpu_timestamp_us):
            raise TraceSummaryError(f"{trace_path}: {category} activity External id {external_id!r} belongs outside {MEASURE_RANGE!r}.")
        stage = _range_stage(cpu_timestamp_us, top_phase_ranges, trace_path=trace_path, subject=f"{category} activity")
        name = event.get("name")
        if not isinstance(name, str) or not name:
            raise TraceSummaryError(f"{trace_path}: {category} activity inside measurement has no name.")
        cuda_groups[(stage, category, name)].append(_number(event, "dur", trace_path=trace_path))

    if physical_event_count == 0:
        raise TraceSummaryError(f"{trace_path}: no kernel/gpu_memcpy/gpu_memset activity found inside {MEASURE_RANGE!r}.")

    rows: list[dict[str, str | int | float]] = []
    range_order = {MEASURE_RANGE: 0, **{name: index + 1 for index, name in enumerate(TOP_PHASES)}, **{name: index + 4 for index, name in enumerate(ATTENTION_RANGES)}}
    for (name, stage, inclusive, notes), durations in sorted(range_groups.items(), key=lambda item: (range_order.get(item[0][0], 99), item[0][1])):
        rows.append(
            _row(
                run_name=run_name,
                row_type="range",
                name=name,
                stage=stage,
                calls=len(durations),
                range_duration_us=sum(durations),
                inclusive=inclusive,
                notes=notes,
            )
        )
    phase_order = {name: index for index, name in enumerate(TOP_PHASES)}
    for (stage, name), durations in sorted(cpu_groups.items(), key=lambda item: (phase_order[item[0][0]], item[0][1])):
        rows.append(
            _row(
                run_name=run_name,
                row_type="cpu_op",
                name=name,
                stage=stage,
                calls=len(durations),
                cpu_time_total_us=sum(durations),
                notes="sum of CPU operation durations; nested operations may overlap",
            )
        )
    for (stage, category, name), durations in sorted(cuda_groups.items(), key=lambda item: (phase_order[item[0][0]], item[0][1], item[0][2])):
        rows.append(
            _row(
                run_name=run_name,
                row_type="cuda_activity",
                name=name,
                stage=stage,
                activity_type=category,
                calls=len(durations),
                cuda_time_total_us=sum(durations),
                kernel_calls=len(durations) if category == "kernel" else 0,
                notes="sum of physical GPU activity durations; not GPU wall-clock time",
            )
        )
    return rows


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except FileNotFoundError as error:
        raise TraceSummaryError(f"Missing profile audit records: {path}") from error
    records: list[dict[str, Any]] = []
    for line_number, line in enumerate(lines, start=1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as error:
            raise TraceSummaryError(f"{path}:{line_number}: invalid JSONL record: {error.msg}.") from error
        if not isinstance(record, dict):
            raise TraceSummaryError(f"{path}:{line_number}: JSONL record must be an object.")
        records.append(record)
    if not records:
        raise TraceSummaryError(f"{path}: no profile audit records found.")
    return records


def _mapping_value(mapping: Mapping[str, Any], key: str, *, record_label: str) -> Any:
    value = mapping.get(key)
    if value is None:
        raise TraceSummaryError(f"{record_label}: missing {key!r}.")
    return value


def run_name_from_record(record: Mapping[str, Any]) -> str:
    """Derive the stable Task 2 run name from a benchmark JSONL record."""

    run_config = record.get("run_config")
    model_config = record.get("model_config")
    if not isinstance(run_config, Mapping) or not isinstance(model_config, Mapping):
        raise TraceSummaryError("Profile audit record must contain object-valued run_config and model_config.")
    model_size = _mapping_value(run_config, "model_size", record_label="Profile audit record")
    context_length = _mapping_value(model_config, "context_length", record_label="Profile audit record")
    mode = _mapping_value(run_config, "mode", record_label="Profile audit record")
    precision = _mapping_value(run_config, "precision", record_label="Profile audit record")
    if not isinstance(model_size, str) or not isinstance(mode, str) or not isinstance(precision, str) or isinstance(context_length, bool) or not isinstance(context_length, int):
        raise TraceSummaryError("Profile audit record has invalid model_size/context_length/mode/precision fields.")
    return f"{model_size}_ctx{context_length}_{mode}_{precision}"


def _safe_command(command: object) -> str:
    """Preserve a reproducible command without retaining absolute local paths."""

    if not isinstance(command, str):
        return ""
    try:
        tokens = shlex.split(command)
    except ValueError:
        return ""

    def clean_token(token: str) -> str:
        if token.startswith("--") and "=" in token:
            option, value = token.split("=", 1)
            if Path(value).is_absolute():
                return f"{option}={Path(value).name}"
            return token
        return Path(token).name if Path(token).is_absolute() else token

    return " ".join(shlex.quote(clean_token(token)) for token in tokens)


def metadata_from_records(records: Sequence[Mapping[str, Any]], *, expected_run_names: Sequence[str]) -> list[dict[str, Any]]:
    """Create public profile metadata from JSONL, preserving the environment."""

    records_by_name: dict[str, Mapping[str, Any]] = {}
    for record in records:
        run_name = run_name_from_record(record)
        if run_name in records_by_name:
            raise TraceSummaryError(f"Profile audit records contain duplicate run {run_name!r}.")
        records_by_name[run_name] = record

    expected = tuple(expected_run_names)
    expected_set = set(expected)
    actual_set = set(records_by_name)
    missing = sorted(expected_set - actual_set)
    unexpected = sorted(actual_set - expected_set)
    if missing or unexpected:
        pieces = []
        if missing:
            pieces.append(f"missing audit records: {', '.join(missing)}")
        if unexpected:
            pieces.append(f"unexpected audit records: {', '.join(unexpected)}")
        raise TraceSummaryError("Profile audit record matrix is not the required six runs (" + "; ".join(pieces) + ").")

    metadata: list[dict[str, Any]] = []
    for run_name in expected:
        record = records_by_name[run_name]
        run_config = record["run_config"]
        model_config = record["model_config"]
        assert isinstance(run_config, Mapping)
        assert isinstance(model_config, Mapping)
        environment = record.get("environment")
        if not isinstance(environment, Mapping):
            raise TraceSummaryError(f"{run_name}: profile audit record has no environment object.")
        required_environment = ("device_name", "torch_version", "cuda_version", "python_version")
        missing_environment = [field for field in required_environment if not isinstance(environment.get(field), str) or not environment.get(field)]
        if missing_environment:
            raise TraceSummaryError(f"{run_name}: profile audit environment is missing {', '.join(missing_environment)}.")
        warmup_steps = _mapping_value(run_config, "warmup_steps", record_label=run_name)
        measurement_steps = _mapping_value(run_config, "measurement_steps", record_label=run_name)
        if isinstance(warmup_steps, bool) or not isinstance(warmup_steps, int) or warmup_steps < 0:
            raise TraceSummaryError(f"{run_name}: invalid warmup_steps in profile audit record.")
        if isinstance(measurement_steps, bool) or not isinstance(measurement_steps, int) or measurement_steps < 1:
            raise TraceSummaryError(f"{run_name}: invalid measurement_steps in profile audit record.")
        if run_config.get("profile_tool") != "torch":
            raise TraceSummaryError(f"{run_name}: profile audit record was not collected with torch.profiler.")
        inside_profiler_steps = 1 if warmup_steps > 0 else 0
        metadata.append(
            {
                "run_name": run_name,
                "model_size": run_config["model_size"],
                "context_length": model_config["context_length"],
                "batch_size": model_config["batch_size"],
                "mode": run_config["mode"],
                "dtype": run_config["precision"],
                "warmup_steps": warmup_steps,
                "warmup_protocol": {
                    "outside_profiler_steps": warmup_steps - inside_profiler_steps,
                    "inside_profiler_steps": inside_profiler_steps,
                },
                "measurement_steps": measurement_steps,
                "tool": "torch.profiler",
                "command": _safe_command(record.get("command")),
                "trace_file": f"{run_name}.json",
                "trace_location": "local_artifacts/profile (not submitted)",
                "summary_file": "trace_summary.csv",
                "environment": {field: environment[field] for field in required_environment},
            }
        )
    return metadata


def _atomic_write_text(path: Path, contents: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True)
    temporary_path = Path(temporary_name)
    try:
        with os.fdopen(file_descriptor, "w", encoding="utf-8", newline="") as output_file:
            output_file.write(contents)
        os.replace(temporary_path, path)
    except BaseException:
        temporary_path.unlink(missing_ok=True)
        raise


def _csv_text(rows: Sequence[Mapping[str, str | int | float]]) -> str:
    from io import StringIO

    output = StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=CSV_FIELDNAMES, extrasaction="raise", lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_trace_summary(trace_path: Path, output_path: Path, *, run_name: str | None = None) -> list[dict[str, str | int | float]]:
    """Write the per-run `--profile-summary` compatibility CSV from a trace."""

    resolved_run_name = trace_path.stem if run_name is None else run_name
    rows = summarize_trace(trace_path, run_name=resolved_run_name)
    _atomic_write_text(output_path, _csv_text(rows))
    return rows


def rebuild_profile_artifacts(
    *,
    trace_dir: Path,
    runs_path: Path,
    output_path: Path,
    metadata_output_path: Path,
    expected_run_names: Sequence[str] = TASK2_RUN_NAMES,
    check_only: bool = False,
) -> ArtifactReport:
    """Validate all six Task 2 traces and atomically rebuild public artifacts."""

    expected = tuple(expected_run_names)
    if len(expected) != len(set(expected)):
        raise TraceSummaryError("Expected profile run names must be unique.")
    records = _read_jsonl(runs_path)
    metadata = metadata_from_records(records, expected_run_names=expected)
    all_rows: list[dict[str, str | int | float]] = []
    for run_name in expected:
        all_rows.extend(summarize_trace(trace_dir / f"{run_name}.json", run_name=run_name, require_all_top_phases=True))
    report = ArtifactReport(run_names=expected, row_count=len(all_rows), metadata_count=len(metadata))
    if not check_only:
        _atomic_write_text(output_path, _csv_text(all_rows))
        _atomic_write_text(metadata_output_path, json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Rebuild Task 2 profile artifacts from saved torch.profiler Chrome traces.")
    parser.add_argument("--trace-dir", type=Path, default=DEFAULT_TRACE_DIR, help="Directory containing the six local Chrome trace JSON files.")
    parser.add_argument("--runs", type=Path, default=DEFAULT_RUNS, help="Task 2 benchmark audit JSONL.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Measurement-only trace summary CSV to write.")
    parser.add_argument("--metadata-output", type=Path, default=DEFAULT_METADATA_OUTPUT, help="Public run_metadata.json to write.")
    parser.add_argument("--check", action="store_true", help="Validate all six traces and audit records without writing output files.")
    return parser


def main(argv: list[str] | None = None) -> ArtifactReport:
    args = build_parser().parse_args(argv)
    try:
        report = rebuild_profile_artifacts(
            trace_dir=args.trace_dir,
            runs_path=args.runs,
            output_path=args.output,
            metadata_output_path=args.metadata_output,
            check_only=args.check,
        )
    except TraceSummaryError as error:
        raise SystemExit(f"trace summary validation failed: {error}") from error
    action = "Validated" if args.check else "Rebuilt"
    print(f"{action} {len(report.run_names)}/6 profile traces, {report.row_count} summary rows, and {report.metadata_count} metadata records.")
    return report


if __name__ == "__main__":
    main()
