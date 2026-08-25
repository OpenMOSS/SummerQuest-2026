#!/usr/bin/env python3
"""Render compact, public-safe PNG assets for the A2-P profiling report.

The raw PyTorch memory snapshots and Chrome traces stay local.  This utility
reads a local snapshot or a lightweight CSV summary and emits only cropped PNG
figures that are suitable for manual review before copying into a public
submission directory.  It never embeds source paths, memory addresses, Python
stack frames, host details, or profiler operator names in the figures.

Examples
--------

Render the three report assets in one command.  ``.pickle`` inputs are accepted
only with ``--allow-pickle`` because Python pickles must be treated as trusted
local inputs::

    uv run python profiling/render_assets.py all \\
        --forward-snapshot local_artifacts/forward.pickle \\
        --train-step-snapshot local_artifacts/train_step.pickle \\
        --trace-summary results/profile/trace_summary.csv \\
        --output-dir results/assets --allow-pickle

Render one memory-history timeline from a JSON snapshot or event/timeline CSV::

    uv run python profiling/render_assets.py memory \\
        --input local_artifacts/train_step.json \\
        --output results/assets/memory_train_step_timeline.png \\
        --workload train_step

Render a CPU/CUDA stage comparison from a lightweight trace summary::

    uv run python profiling/render_assets.py profile \\
        --input results/profile/trace_summary.csv \\
        --output results/assets/profile_stage_times.png

Render a CPU ``record_function`` stage timeline from a local Chrome trace::

    uv run python profiling/render_assets.py timeline \
        --input local_artifacts/train_step_trace.json \
        --output results/assets/profile_timeline.png

For a PyTorch allocator snapshot, the memory chart reconstructs *recorded
active allocation bytes* from ``device_traces``.  A memory-history snapshot
does not normally include a timestamp, so its x-axis is event index rather than
wall-clock time.  When memory history was enabled after warm-up, the curve is
relative to allocations observed after that boundary; it is not relabelled as
the allocator's full ``active_bytes`` counter.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import pickle
import re
import sys
import tempfile
from collections import OrderedDict, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib


# A non-interactive backend is required on CUDA servers and CI workers without
# a desktop session.  It must be selected before importing pyplot.
matplotlib.use("Agg")

from matplotlib import pyplot as plt


MEBIBYTE = 1024**2
DEFAULT_MAX_POINTS = 6_000
PNG_METADATA = {
    "Title": "A2-P profiling visualization",
    "Author": "",
    "Description": "Public-safe derived profiling figure; raw artifacts remain local.",
    "Software": "matplotlib",
}

WORKLOAD_LABELS = {
    "forward": "Forward-only",
    "train_step": "Train step",
}

STAGE_SPECS = (
    ("forward", "Forward"),
    ("backward", "Backward"),
    ("optimizer", "Optimizer"),
    ("attention/scores", "Attention scores"),
    ("attention/softmax", "Attention softmax"),
    ("attention/value", "Attention value"),
)

# These are intentionally exact trace names. The parser below accepts only
# CPU/GPU annotation complete events with one of these names, so a public
# figure cannot accidentally surface operator names, kernel names, paths, or
# trace metadata.
TRACE_LANE_SPECS = (
    ("profile/measure", "Measurement step", "#546e7a"),
    ("forward", "Forward", "#1976d2"),
    ("attention/scores", "Attention scores", "#7b1fa2"),
    ("attention/softmax", "Attention softmax", "#ab47bc"),
    ("attention/value", "Attention value", "#8e24aa"),
    ("backward", "Backward", "#d84315"),
    ("optimizer", "Optimizer", "#00897b"),
)
TRACE_STAGE_TO_LANE = {
    stage: (index, label, color)
    for index, (stage, label, color) in enumerate(TRACE_LANE_SPECS)
}

ALLOC_ACTIONS = {"alloc", "allocate", "allocation", "malloc"}
FREE_COMPLETED_ACTIONS = {"free", "free_completed", "dealloc", "deallocate", "release"}
OOM_ACTIONS = {"oom", "out_of_memory", "outofmemory"}

ABSOLUTE_PATH_RE = re.compile(r"(?<![\w<])/(?:[^\s:'\"()\[\],]+/)*[^\s:'\"()\[\],]+")
WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s:'\"()\[\],]+\\)*[^\s:'\"()\[\],]+")
IP_ADDRESS_RE = re.compile(r"\b(?:\d{1,3}\.){3}\d{1,3}\b")
UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-(?:[0-9a-fA-F]{4}-){3}[0-9a-fA-F]{12}\b")


@dataclass(frozen=True)
class TimelinePoint:
    """One public-safe sample of allocation state at a trace event index."""

    event_index: int
    active_bytes: int


@dataclass(frozen=True)
class AllocationTimeline:
    """A reconstructed timeline without memory addresses or stack-frame data."""

    points: tuple[TimelinePoint, ...]
    event_count: int
    allocation_count: int
    completed_free_count: int
    unmatched_free_count: int
    reused_address_count: int
    oom_event_indices: tuple[int, ...]
    device_index: int | None
    source_scope: str


@dataclass(frozen=True)
class StageTiming:
    """CPU and CUDA time for one public profiler stage, in microseconds."""

    stage: str
    label: str
    cpu_total_us: float
    cuda_total_us: float
    contributing_runs: int


@dataclass(frozen=True)
class TraceRange:
    """One public-safe stage range in relative milliseconds."""

    stage: str
    lane_label: str
    lane_index: int
    color: str
    start_ms: float
    duration_ms: float


@dataclass(frozen=True)
class TraceLane:
    """A display-only lane made from a whitelisted Chrome trace category."""

    label: str
    ranges: tuple[TraceRange, ...]


@dataclass(frozen=True)
class RenderResult:
    """Small, non-sensitive result used for command-line status output."""

    kind: str
    displayed_points: int | None = None
    stages: int | None = None
    ranges: int | None = None


class RestrictedSnapshotUnpickler(pickle.Unpickler):
    """Reject globals when reading a trusted allocator snapshot pickle.

    PyTorch's allocator snapshots are plain dictionaries/lists/numbers.  This
    deliberately permits only ``OrderedDict`` in addition to pickle's native
    collection opcodes, which prevents arbitrary class construction.  The
    ``--allow-pickle`` opt-in remains necessary because pickle inputs should
    always be treated as trusted local artifacts.
    """

    def find_class(self, module: str, name: str) -> Any:
        if (module, name) == ("collections", "OrderedDict"):
            return OrderedDict
        raise pickle.UnpicklingError("snapshot pickle contains a disallowed global")


def nonnegative_int(value: str) -> int:
    """Argparse validator for device indices and nonnegative limits."""

    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("必须是非负整数")
    return parsed


def positive_int(value: str) -> int:
    """Argparse validator for point limits."""

    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("必须是正整数")
    return parsed


def public_error_text(value: object, limit: int = 360) -> str:
    """Redact values that must never be echoed by a public-artifact utility."""

    text = " ".join(str(value).split())
    text = IP_ADDRESS_RE.sub("<ip>", text)
    text = UUID_RE.sub("<id>", text)
    text = ABSOLUTE_PATH_RE.sub("<path>", text)
    text = WINDOWS_PATH_RE.sub("<path>", text)
    return text[:limit]


def require_regular_file(path: Path, kind: str) -> None:
    """Fail clearly without echoing a local path or input basename."""

    if not path.is_file():
        raise FileNotFoundError(f"找不到所需的{kind}输入文件，或输入不是普通文件")


def validate_png_output(path: Path) -> None:
    """Keep the public asset contract explicit and avoid accidental non-PNG output."""

    if path.suffix.lower() != ".png":
        raise ValueError("公开图像输出必须使用 .png 扩展名")
    if path.exists() and path.is_dir():
        raise ValueError("图像输出目标不能是目录")


def normalize_key(value: object) -> str:
    """Normalize schema aliases without altering free-form values shown nowhere."""

    return re.sub(r"[^a-z0-9]+", "_", str(value).strip().lower()).strip("_")


def normalize_action(value: object) -> str:
    return normalize_key(value)


def normalized_field_lookup(fieldnames: Sequence[str]) -> dict[str, str]:
    return {normalize_key(field): field for field in fieldnames}


def parse_byte_count(value: object) -> int | None:
    """Parse byte counts from snapshots and compact CSV values conservatively."""

    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        if not math.isfinite(float(value)) or float(value) < 0:
            return None
        return int(round(float(value)))

    compact = str(value).strip().replace(",", "")
    if not compact or compact.lower() in {"na", "n/a", "none", "null", "-", "--"}:
        return None
    match = re.fullmatch(r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(b|bytes?|kib|mib|gib|kb|mb|gb)?", compact, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    if not math.isfinite(number) or number < 0:
        return None
    unit = (match.group(2) or "b").lower()
    multipliers = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kib": 1024,
        "mib": MEBIBYTE,
        "gib": 1024**3,
        "kb": 1_000,
        "mb": 1_000_000,
        "gb": 1_000_000_000,
    }
    return int(round(number * multipliers[unit]))


def parse_microseconds(value: object) -> float | None:
    """Parse raw microseconds and common profiler duration spellings."""

    if value is None:
        return None
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        parsed = float(value)
        return parsed if math.isfinite(parsed) and parsed >= 0 else None

    compact = str(value).strip().replace(",", "")
    if not compact or compact.lower() in {"na", "n/a", "none", "null", "-", "--"}:
        return None
    match = re.fullmatch(r"([+]?(?:\d+(?:\.\d*)?|\.\d+)(?:e[+-]?\d+)?)\s*(ns|us|µs|ms|s)?", compact, re.IGNORECASE)
    if match is None:
        return None
    number = float(match.group(1))
    if not math.isfinite(number) or number < 0:
        return None
    unit = (match.group(2) or "us").lower()
    multipliers = {"ns": 0.001, "us": 1.0, "µs": 1.0, "ms": 1_000.0, "s": 1_000_000.0}
    return number * multipliers[unit]


def parse_event_index(value: object, fallback: int) -> int:
    """Return a nonnegative CSV event index, falling back to row order."""

    parsed = parse_byte_count(value)
    return fallback if parsed is None else parsed


def event_address(event: Mapping[str, object], fallback: int) -> str:
    """Use an opaque internal key only; addresses are never placed in output."""

    for name in ("addr", "address", "ptr", "pointer"):
        value = event.get(name)
        if value is not None and str(value).strip():
            return f"address:{str(value).strip()}"
    return f"event:{fallback}"


def event_size(event: Mapping[str, object]) -> int | None:
    for name in ("size", "size_bytes", "allocated_size", "bytes"):
        parsed = parse_byte_count(event.get(name))
        if parsed is not None:
            return parsed
    return None


def normalize_event(event: Mapping[str, object]) -> dict[str, object]:
    """Accept JSON/CSV aliases while retaining only allocator-event fields."""

    lookup = {normalize_key(key): key for key in event}

    def read(*aliases: str) -> object | None:
        for alias in aliases:
            key = lookup.get(normalize_key(alias))
            if key is not None:
                return event.get(key)
        return None

    return {
        "action": read("action", "event", "kind", "type"),
        "addr": read("addr", "address", "ptr", "pointer"),
        "size": read("size", "size_bytes", "allocated_size", "bytes"),
    }


def reconstruct_allocation_timeline(events: Sequence[Mapping[str, object]], device_index: int | None, source_scope: str) -> AllocationTimeline:
    """Reconstruct allocations observed after the memory-history boundary.

    ``free_requested`` is intentionally not a negative delta: PyTorch still
    considers the allocation active until ``free_completed``.  Segment events
    track cached/reserved allocator segments and are deliberately excluded from
    the active-allocation curve.
    """

    if not events:
        raise ValueError("snapshot 没有可用的 device_traces allocator event")

    active_by_address: dict[str, int] = {}
    active_bytes = 0
    allocation_count = 0
    completed_free_count = 0
    unmatched_free_count = 0
    reused_address_count = 0
    oom_indices: list[int] = []
    points = [TimelinePoint(event_index=0, active_bytes=0)]

    for event_index, raw_event in enumerate(events, start=1):
        event = normalize_event(raw_event)
        action = normalize_action(event.get("action"))
        size = event_size(event)
        address = event_address(event, event_index)

        if action in ALLOC_ACTIONS:
            if size is not None and size > 0:
                old_size = active_by_address.get(address)
                if old_size is not None:
                    active_bytes -= old_size
                    reused_address_count += 1
                active_by_address[address] = size
                active_bytes += size
                allocation_count += 1
        elif action in FREE_COMPLETED_ACTIONS:
            allocated_size = active_by_address.pop(address, None)
            if allocated_size is None:
                # Memory history may begin after warm-up, so an unmatched free
                # can belong to a baseline allocation.  Do not subtract it from
                # the post-boundary curve and accidentally claim full active bytes.
                unmatched_free_count += 1
            else:
                active_bytes -= allocated_size
                completed_free_count += 1
        elif action in OOM_ACTIONS:
            oom_indices.append(event_index)

        # A corrupt or partial event must never yield a negative public memory
        # count.  This is a guard only; normal allocator event pairs preserve it.
        active_bytes = max(active_bytes, 0)
        points.append(TimelinePoint(event_index=event_index, active_bytes=active_bytes))

    if allocation_count == 0:
        raise ValueError("snapshot event 中没有可用于重构 timeline 的 alloc 记录")
    return AllocationTimeline(
        points=tuple(points),
        event_count=len(events),
        allocation_count=allocation_count,
        completed_free_count=completed_free_count,
        unmatched_free_count=unmatched_free_count,
        reused_address_count=reused_address_count,
        oom_event_indices=tuple(oom_indices),
        device_index=device_index,
        source_scope=source_scope,
    )


def device_index_from_value(value: object) -> int | None:
    """Read common CSV device labels such as ``0`` and ``cuda:0``."""

    if value is None:
        return None
    compact = str(value).strip().lower()
    if not compact:
        return None
    if compact.isdigit():
        return int(compact)
    match = re.fullmatch(r"cuda:(\d+)", compact)
    return int(match.group(1)) if match is not None else None


def select_snapshot_events(snapshot: object, requested_device: int | None) -> tuple[list[Mapping[str, object]], int]:
    """Select exactly one nonempty ``device_traces`` stream from a snapshot."""

    if isinstance(snapshot, list):
        if all(isinstance(event, Mapping) for event in snapshot):
            return [dict(event) for event in snapshot], 0
        raise ValueError("JSON snapshot 列表必须由 allocator event 对象组成")
    if not isinstance(snapshot, Mapping):
        raise ValueError("snapshot 顶层必须是对象，且包含 device_traces")

    traces = snapshot.get("device_traces", snapshot.get("traces", snapshot.get("events")))
    available: dict[int, list[Mapping[str, object]]] = {}
    if isinstance(traces, Mapping):
        for raw_device, raw_events in traces.items():
            device = device_index_from_value(raw_device)
            if device is None or not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
                continue
            events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
            if events:
                available[device] = events
    elif isinstance(traces, Sequence) and not isinstance(traces, (str, bytes)):
        # ``events`` may be a direct list of event mappings, whereas
        # ``device_traces`` is normally a list indexed by CUDA device id.
        if all(isinstance(item, Mapping) for item in traces):
            available[0] = [dict(item) for item in traces]
        else:
            for device, raw_events in enumerate(traces):
                if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
                    continue
                events = [dict(event) for event in raw_events if isinstance(event, Mapping)]
                if events:
                    available[device] = events

    if not available:
        raise ValueError("snapshot 不包含非空的 device_traces event 序列")
    if requested_device is None:
        if len(available) != 1:
            raise ValueError("snapshot 含多个有数据的设备；请显式传入 --device")
        selected_device = next(iter(available))
    else:
        selected_device = requested_device
    if selected_device not in available:
        raise ValueError("请求的 CUDA device 没有可用的 allocator event")
    return available[selected_device], selected_device


def load_pickle_snapshot(path: Path, allow_pickle: bool) -> object:
    """Load a trusted, restricted PyTorch allocator snapshot pickle."""

    if not allow_pickle:
        raise ValueError("读取 .pickle/.pkl snapshot 需要显式传入 --allow-pickle；只允许可信的本地文件")
    try:
        with path.open("rb") as handle:
            return RestrictedSnapshotUnpickler(handle).load()
    except (OSError, pickle.UnpicklingError, EOFError, ValueError) as error:
        raise ValueError(f"无法读取可信 snapshot pickle：{public_error_text(error)}") from None


def load_json_snapshot(path: Path) -> object:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError(f"无法读取 snapshot JSON：{public_error_text(error)}") from None


def load_chrome_trace_lanes(path: Path) -> list[TraceLane]:
    """Extract only whitelisted CPU/GPU annotation ranges from one Chrome trace.

    The public output deliberately keeps no raw timestamps, process/thread IDs,
    external IDs, trace metadata, kernel names, or event arguments. CPU and GPU
    are separate asynchronous lanes and are never added together.
    """

    require_regular_file(path, "Chrome trace")
    if path.suffix.lower() != ".json":
        raise ValueError("Chrome trace 输入必须是 .json 文件")
    document = load_json_snapshot(path)
    if not isinstance(document, Mapping):
        raise ValueError("Chrome trace 顶层必须是对象")
    raw_events = document.get("traceEvents")
    if not isinstance(raw_events, Sequence) or isinstance(raw_events, (str, bytes)):
        raise ValueError("Chrome trace 缺少 traceEvents 事件列表")

    cpu_candidates: list[tuple[int, str, float, float]] = []
    gpu_candidates: list[tuple[int, str, float, float]] = []
    for original_index, raw_event in enumerate(raw_events):
        if not isinstance(raw_event, Mapping) or raw_event.get("ph") != "X":
            continue
        category = raw_event.get("cat")
        if category not in {"user_annotation", "gpu_user_annotation"}:
            continue
        stage = raw_event.get("name")
        if not isinstance(stage, str) or stage not in TRACE_STAGE_TO_LANE:
            continue
        timestamp = raw_event.get("ts")
        duration = raw_event.get("dur")
        if isinstance(timestamp, bool) or isinstance(duration, bool):
            continue
        if not isinstance(timestamp, (int, float)) or not isinstance(duration, (int, float)):
            continue
        start_us = float(timestamp)
        duration_us = float(duration)
        if not (math.isfinite(start_us) and math.isfinite(duration_us) and start_us >= 0 and duration_us > 0):
            continue
        candidate = (original_index, stage, start_us, duration_us)
        if category == "user_annotation":
            cpu_candidates.append(candidate)
        else:
            gpu_candidates.append(candidate)

    measure_candidates = [candidate for candidate in cpu_candidates if candidate[1] == "profile/measure"]
    if len(measure_candidates) != 1:
        raise ValueError("Chrome trace 必须恰好包含一个有效的 profile/measure CPU 范围")
    _, _, measure_start_us, measure_duration_us = measure_candidates[0]
    measure_end_us = measure_start_us + measure_duration_us

    def make_ranges(
        candidates: Sequence[tuple[int, str, float, float]],
        *,
        include_measure: bool,
    ) -> list[TraceRange]:
        converted: list[tuple[float, int, float, str, int]] = []
        for original_index, stage, start_us, duration_us in candidates:
            if not include_measure and stage == "profile/measure":
                continue
            if start_us < measure_start_us or start_us + duration_us > measure_end_us + 1e-3:
                continue
            lane_index, lane_label, color = TRACE_STAGE_TO_LANE[stage]
            converted.append(
                (
                    (start_us - measure_start_us) / 1_000.0,
                    lane_index,
                    duration_us / 1_000.0,
                    stage,
                    original_index,
                )
            )
        converted.sort(key=lambda item: (item[0], item[1], -item[2], item[3], item[4]))
        return [
            TraceRange(
                stage=stage,
                lane_label=TRACE_STAGE_TO_LANE[stage][1],
                lane_index=lane_index,
                color=TRACE_STAGE_TO_LANE[stage][2],
                start_ms=start_ms,
                duration_ms=duration_ms,
            )
            for start_ms, lane_index, duration_ms, stage, _ in converted
        ]

    cpu_ranges = make_ranges(cpu_candidates, include_measure=True)
    gpu_ranges = make_ranges(gpu_candidates, include_measure=False)
    if len(cpu_ranges) <= 1:
        raise ValueError("Chrome trace 没有 profile/measure 内的 CPU 阶段范围")
    lanes = [TraceLane(label="Host CPU", ranges=tuple(cpu_ranges))]
    if gpu_ranges:
        lanes.append(TraceLane(label="GPU stream", ranges=tuple(gpu_ranges)))
    return lanes


def load_chrome_trace_ranges(path: Path) -> list[TraceRange]:
    """Backward-compatible CPU-only wrapper for earlier callers."""

    return list(load_chrome_trace_lanes(path)[0].ranges)


def read_csv_rows(path: Path, kind: str) -> tuple[list[str], list[dict[str, str]]]:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.DictReader(handle)
            if not reader.fieldnames:
                raise ValueError(f"{kind} CSV 缺少 header")
            fields = [str(field) for field in reader.fieldnames]
            return fields, [dict(row) for row in reader]
    except (OSError, csv.Error, UnicodeError) as error:
        raise ValueError(f"无法读取{kind} CSV：{public_error_text(error)}") from None


def csv_device_rows(rows: Sequence[Mapping[str, str]], lookup: Mapping[str, str], requested_device: int | None) -> tuple[list[dict[str, str]], int | None]:
    """Filter an event/timeline CSV to a single device only when it declares one."""

    device_field = next((lookup.get(normalize_key(alias)) for alias in ("device", "device_index", "cuda_device")), None)
    if device_field is None:
        return [dict(row) for row in rows], None

    by_device: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        device = device_index_from_value(row.get(device_field))
        if device is not None:
            by_device[device].append(dict(row))
    if not by_device:
        raise ValueError("memory CSV 声明了 device 列，但没有可解析的 CUDA device")
    if requested_device is None:
        if len(by_device) != 1:
            raise ValueError("memory CSV 含多个设备；请显式传入 --device")
        selected_device = next(iter(by_device))
    else:
        selected_device = requested_device
    selected_rows = by_device.get(selected_device)
    if not selected_rows:
        raise ValueError("请求的 CUDA device 在 memory CSV 中没有记录")
    return selected_rows, selected_device


def load_timeline_csv(path: Path, requested_device: int | None) -> AllocationTimeline:
    """Load either allocator event rows or precomputed active-byte samples."""

    fields, rows = read_csv_rows(path, "memory timeline")
    lookup = normalized_field_lookup(fields)
    selected_rows, selected_device = csv_device_rows(rows, lookup, requested_device)
    action_field = next((lookup.get(normalize_key(alias)) for alias in ("action", "event", "kind", "type")), None)
    active_field = next(
        (
            lookup.get(normalize_key(alias))
            for alias in ("active_allocation_bytes", "active_bytes", "recorded_active_bytes", "active_mib", "active_allocation_mib")
            if lookup.get(normalize_key(alias)) is not None
        ),
        None,
    )

    if action_field is not None:
        events = [dict(row) for row in selected_rows]
        return reconstruct_allocation_timeline(events, selected_device, source_scope="recorded")
    if active_field is None:
        raise ValueError("memory CSV 需要 action/size 事件列，或 active_bytes/active_mib timeline 列")

    active_is_mib = normalize_key(active_field).endswith("mib")
    index_field = next((lookup.get(normalize_key(alias)) for alias in ("event_index", "index", "step", "event_id")), None)
    points: list[TimelinePoint] = []
    for fallback_index, row in enumerate(selected_rows):
        raw_value = row.get(active_field)
        if active_is_mib:
            try:
                active_bytes = int(round(float(str(raw_value).replace(",", "")) * MEBIBYTE))
            except (TypeError, ValueError):
                continue
        else:
            active_bytes = parse_byte_count(raw_value)
            if active_bytes is None:
                continue
        event_index = parse_event_index(row.get(index_field) if index_field is not None else None, fallback_index)
        points.append(TimelinePoint(event_index=event_index, active_bytes=max(active_bytes, 0)))
    if not points:
        raise ValueError("memory CSV 没有可解析的 active-byte timeline 数据")
    points.sort(key=lambda point: point.event_index)
    return AllocationTimeline(
        points=tuple(points),
        event_count=max(point.event_index for point in points),
        allocation_count=0,
        completed_free_count=0,
        unmatched_free_count=0,
        reused_address_count=0,
        oom_event_indices=(),
        device_index=selected_device,
        source_scope="reported",
    )


def load_memory_timeline(path: Path, requested_device: int | None, allow_pickle: bool) -> AllocationTimeline:
    """Load supported local input formats without copying their raw content."""

    require_regular_file(path, "memory snapshot")
    suffix = path.suffix.lower()
    if suffix in {".pickle", ".pkl"}:
        snapshot = load_pickle_snapshot(path, allow_pickle)
        events, device_index = select_snapshot_events(snapshot, requested_device)
        return reconstruct_allocation_timeline(events, device_index, source_scope="recorded")
    if suffix == ".json":
        snapshot = load_json_snapshot(path)
        events, device_index = select_snapshot_events(snapshot, requested_device)
        return reconstruct_allocation_timeline(events, device_index, source_scope="recorded")
    if suffix == ".csv":
        return load_timeline_csv(path, requested_device)
    raise ValueError("memory 输入只支持 .pickle、.pkl、.json 或 .csv")


def downsample_timeline(points: Sequence[TimelinePoint], max_points: int) -> list[TimelinePoint]:
    """Preserve endpoints and local extrema without rendering millions of vertices."""

    if len(points) <= max_points:
        return list(points)
    if max_points < 8:
        raise ValueError("--max-points 至少应为 8，才能保留峰值与端点")

    bucket_count = max(1, max_points // 4)
    bucket_width = math.ceil(len(points) / bucket_count)
    selected: dict[int, TimelinePoint] = {}
    for start in range(0, len(points), bucket_width):
        bucket = points[start : start + bucket_width]
        candidates = (
            bucket[0],
            min(bucket, key=lambda point: point.active_bytes),
            max(bucket, key=lambda point: point.active_bytes),
            bucket[-1],
        )
        for point in candidates:
            selected[point.event_index] = point
    selected[points[0].event_index] = points[0]
    selected[points[-1].event_index] = points[-1]
    return [selected[index] for index in sorted(selected)]


def format_mib(value: int | float) -> str:
    return f"{float(value) / MEBIBYTE:.1f} MiB"


def save_public_png(figure: plt.Figure, output: Path) -> None:
    """Atomically save a tightly cropped PNG with non-sensitive metadata only."""

    validate_png_output(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=".render-assets-", suffix=".png", dir=output.parent)
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    try:
        figure.savefig(
            temporary_path,
            format="png",
            dpi=180,
            bbox_inches="tight",
            pad_inches=0.06,
            facecolor="white",
            metadata=PNG_METADATA,
        )
        os.replace(temporary_path, output)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
        plt.close(figure)


def render_memory_timeline(timeline: AllocationTimeline, workload: str, output: Path, max_points: int) -> RenderResult:
    """Render a cropped active-allocation PNG without raw allocator details."""

    displayed = downsample_timeline(timeline.points, max_points)
    x_values = [point.event_index for point in displayed]
    y_values = [point.active_bytes / MEBIBYTE for point in displayed]
    peak = max(timeline.points, key=lambda point: point.active_bytes)

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 4.75))
        axis.step(x_values, y_values, where="post", color="#1769aa", linewidth=1.45, label="Allocation delta")
        axis.fill_between(x_values, y_values, step="post", color="#42a5f5", alpha=0.23)
        axis.scatter([peak.event_index], [peak.active_bytes / MEBIBYTE], color="#b71c1c", s=24, zorder=4)

        # An OOM marker is meaningful but its raw request/error text is not public.
        if timeline.oom_event_indices:
            marker_index = timeline.oom_event_indices[0]
            marker_y = max(y_values) * 0.96 if max(y_values) > 0 else 0.0
            axis.axvline(marker_index, color="#b71c1c", linewidth=1.0, linestyle="--", alpha=0.75)
            axis.annotate("OOM event", xy=(marker_index, marker_y), xytext=(4, 2), textcoords="offset points", color="#b71c1c", fontsize=8)

        label = WORKLOAD_LABELS[workload]
        axis.set_title(f"{label}: memory-history allocation timeline", loc="left", fontweight="bold")
        axis.set_xlabel("Memory-history event index")
        axis.set_ylabel("Recorded allocation delta (MiB)")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.set_xlim(min(x_values), max(x_values) if max(x_values) > min(x_values) else min(x_values) + 1)
        axis.set_ylim(bottom=0)
        axis.legend(loc="upper left", frameon=False)
        peak_text = f"Maximum recorded delta: {format_mib(peak.active_bytes)} at event {peak.event_index:,}"
        axis.text(0.995, 0.965, peak_text, transform=axis.transAxes, ha="right", va="top", fontsize=9, color="#263238")

        if timeline.source_scope == "recorded":
            note = "History starts after warm-up; this is allocation delta after that boundary, not full allocator active_bytes."
        else:
            note = "Curve uses active-byte samples supplied by the reviewed CSV input."
        figure.text(0.125, 0.012, note, ha="left", va="bottom", fontsize=8.1, color="#455a64")
        figure.subplots_adjust(left=0.115, right=0.975, top=0.88, bottom=0.15)
        save_public_png(figure, output)
    return RenderResult(kind="memory", displayed_points=len(displayed))


def render_trace_timeline(lanes: Sequence[TraceLane], output: Path) -> RenderResult:
    """Render separate public CPU/GPU lanes from whitelisted trace annotations."""

    if not lanes or not any(lane.ranges for lane in lanes):
        raise ValueError("没有可绘制的 Chrome trace 阶段范围")
    max_end_ms = max(item.start_ms + item.duration_ms for lane in lanes for item in lane.ranges)
    if max_end_ms <= 0:
        raise ValueError("Chrome trace 阶段范围没有正的相对时长")

    positions: list[float] = []
    labels: list[str] = []
    row_by_lane_stage: dict[tuple[int, int], float] = {}
    next_row = 0.0
    for lane_offset, lane in enumerate(lanes):
        present_stages = sorted({item.lane_index for item in lane.ranges})
        for stage_index in present_stages:
            row_by_lane_stage[(lane_offset, stage_index)] = next_row
            labels.append(f"{lane.label}: {TRACE_LANE_SPECS[stage_index][1]}")
            positions.append(next_row)
            next_row += 1.0
        next_row += 0.75

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure_height = max(4.8, 0.48 * len(positions) + 1.75)
        figure, axis = plt.subplots(figsize=(10.5, figure_height))
        for lane_offset, lane in enumerate(lanes):
            for item in lane.ranges:
                y = row_by_lane_stage[(lane_offset, item.lane_index)]
                axis.broken_barh(
                    [(item.start_ms, item.duration_ms)],
                    (y - 0.34, 0.68),
                    facecolors=item.color,
                    edgecolors="white",
                    linewidth=0.55,
                    alpha=0.91,
                )
                if item.duration_ms < 0.01:
                    axis.plot(item.start_ms, y, marker="|", markersize=8, color=item.color, markeredgewidth=1.0)

        axis.set_yticks(positions, labels)
        axis.invert_yaxis()
        axis.set_xlabel("Relative time from measurement start (ms)")
        axis.set_title("Profile stage timeline", loc="left", fontweight="bold")
        axis.grid(axis="x", color="#d9d9d9", linewidth=0.7, alpha=0.85)
        axis.set_axisbelow(True)
        axis.set_xlim(0, max_end_ms * 1.025)
        axis.set_ylim(max(positions) + 0.5, -0.5)
        note = "Whitelisted annotation ranges only. CPU and GPU lanes are asynchronous; nested ranges are inclusive and not summed."
        figure.text(0.125, 0.012, note, ha="left", va="bottom", fontsize=8.1, color="#455a64")
        figure.subplots_adjust(left=0.34, right=0.975, top=0.89, bottom=0.15)
        save_public_png(figure, output)
    return RenderResult(kind="trace_timeline", ranges=sum(len(lane.ranges) for lane in lanes))


def canonical_stage(value: object) -> str | None:
    """Recognize only explicit public stage names; never plot raw operator names."""

    compact = str(value).strip().lower()
    normalized = re.sub(r"[\\_\-\s]+", "/", compact)
    aliases = {
        "forward": "forward",
        "backward": "backward",
        "optimizer": "optimizer",
        "attention/scores": "attention/scores",
        "attention/score": "attention/scores",
        "attention/softmax": "attention/softmax",
        "attention/value": "attention/value",
        "attention/values": "attention/value",
    }
    return aliases.get(normalized)


def aggregate_values(values: Sequence[float], mode: str) -> float:
    if not values:
        return 0.0
    if mode == "sum":
        return sum(values)
    if mode == "max":
        return max(values)
    return sum(values) / len(values)


def extract_stage_timings(path: Path, requested_run: str | None, aggregation: str) -> list[StageTiming]:
    """Read trace_summary CSVs from benchmark.py or summarize.py.

    Exact ``record_function``/NVTX stage rows are preferred.  If a normalized
    summary lacks those explicit rows, its canonical ``stage`` column is used as
    a fallback.  This avoids showing raw kernels/operators and prevents a direct
    stage row from being double-counted with its nested operations.
    """

    require_regular_file(path, "profile trace summary")
    if path.suffix.lower() != ".csv":
        raise ValueError("profile 输入必须是 trace_summary CSV")
    fields, rows = read_csv_rows(path, "profile trace summary")
    lookup = normalized_field_lookup(fields)
    op_field = next((lookup.get(normalize_key(alias)) for alias in ("op_name", "operator", "op", "name", "key")), None)
    stage_field = next((lookup.get(normalize_key(alias)) for alias in ("stage", "stage_range", "range", "nvtx_range")), None)
    if op_field is None and stage_field is None:
        raise ValueError("trace_summary CSV 需要 op_name 或 stage 列")

    run_field = next((lookup.get(normalize_key(alias)) for alias in ("run_name", "run", "profile_name", "trace_name")), None)
    cpu_field = next(
        (
            lookup.get(normalize_key(alias))
            for alias in ("cpu_total_time_us", "cpu_total_us", "cpu_time_total", "cpu_time")
            if lookup.get(normalize_key(alias)) is not None
        ),
        None,
    )
    cuda_field = next(
        (
            lookup.get(normalize_key(alias))
            for alias in ("cuda_total_time_us", "cuda_total_us", "cuda_time_total", "cuda_time", "device_time_total", "gpu_time_total")
            if lookup.get(normalize_key(alias)) is not None
        ),
        None,
    )
    if cpu_field is None and cuda_field is None:
        raise ValueError("trace_summary CSV 缺少 CPU/CUDA 累计时间列")

    direct: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    fallback: dict[str, dict[str, list[tuple[float, float]]]] = defaultdict(lambda: defaultdict(list))
    all_runs: set[str] = set()
    for row in rows:
        run = row.get(run_field, "profile") if run_field is not None else "profile"
        run = str(run).strip() or "profile"
        all_runs.add(run)
        cpu_us = parse_microseconds(row.get(cpu_field)) if cpu_field is not None else 0.0
        cuda_us = parse_microseconds(row.get(cuda_field)) if cuda_field is not None else 0.0
        cpu_us = 0.0 if cpu_us is None else cpu_us
        cuda_us = 0.0 if cuda_us is None else cuda_us
        direct_stage = canonical_stage(row.get(op_field, "")) if op_field is not None else None
        fallback_stage = canonical_stage(row.get(stage_field, "")) if stage_field is not None else None
        if direct_stage is not None:
            direct[run][direct_stage].append((cpu_us, cuda_us))
        elif fallback_stage is not None:
            fallback[run][fallback_stage].append((cpu_us, cuda_us))

    if requested_run is not None:
        selected_runs = [requested_run] if requested_run in all_runs else []
        if not selected_runs:
            raise ValueError("所请求的 profile run 不存在于 trace_summary CSV")
    else:
        selected_runs = sorted(all_runs)
    if not selected_runs:
        raise ValueError("trace_summary CSV 没有 profile run 记录")

    results: list[StageTiming] = []
    for stage, label in STAGE_SPECS:
        per_run_cpu: list[float] = []
        per_run_cuda: list[float] = []
        for run in selected_runs:
            samples = direct[run].get(stage)
            if not samples:
                samples = fallback[run].get(stage)
            if not samples:
                continue
            per_run_cpu.append(sum(cpu for cpu, _ in samples))
            per_run_cuda.append(sum(cuda for _, cuda in samples))
        if per_run_cpu or per_run_cuda:
            results.append(
                StageTiming(
                    stage=stage,
                    label=label,
                    cpu_total_us=aggregate_values(per_run_cpu, aggregation),
                    cuda_total_us=aggregate_values(per_run_cuda, aggregation),
                    contributing_runs=max(len(per_run_cpu), len(per_run_cuda)),
                )
            )
    if not results:
        raise ValueError("trace_summary CSV 没有 forward/backward/optimizer/attention 阶段记录")
    if all(timing.cpu_total_us == 0 and timing.cuda_total_us == 0 for timing in results):
        raise ValueError("trace_summary 的目标阶段时间均为 0，无法生成有意义的时间图")
    return results


def render_stage_bars(timings: Sequence[StageTiming], output: Path, aggregation: str) -> RenderResult:
    """Render a compact grouped CPU/CUDA bar chart from public stage labels."""

    labels = [timing.label for timing in timings]
    cpu_ms = [timing.cpu_total_us / 1_000.0 for timing in timings]
    cuda_ms = [timing.cuda_total_us / 1_000.0 for timing in timings]
    positions = list(range(len(timings)))
    width = 0.36
    contributing_runs = max(timing.contributing_runs for timing in timings)
    aggregation_text = {"mean": "Mean", "sum": "Sum", "max": "Maximum"}[aggregation]

    with plt.rc_context(
        {
            "font.size": 10,
            "axes.titlesize": 14,
            "axes.labelsize": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    ):
        figure, axis = plt.subplots(figsize=(9.2, 5.0))
        cpu_positions = [position - width / 2 for position in positions]
        cuda_positions = [position + width / 2 for position in positions]
        axis.bar(cpu_positions, cpu_ms, width=width, label="CPU total", color="#78909c")
        axis.bar(cuda_positions, cuda_ms, width=width, label="CUDA total", color="#ef6c00")
        axis.set_xticks(positions, labels, rotation=18, ha="right")
        axis.set_ylabel("Cumulative time (ms)")
        axis.set_title("Profile stage timing", loc="left", fontweight="bold")
        axis.grid(axis="y", color="#d9d9d9", linewidth=0.7, alpha=0.8)
        axis.legend(frameon=False, ncols=2, loc="upper left")
        subtitle = f"{aggregation_text} across up to {contributing_runs} captured run(s). CPU/CUDA totals are inclusive and may overlap across nested stages."
        figure.text(0.125, 0.012, subtitle, ha="left", va="bottom", fontsize=8.1, color="#455a64")
        figure.subplots_adjust(left=0.105, right=0.975, top=0.88, bottom=0.24)
        save_public_png(figure, output)
    return RenderResult(kind="profile", stages=len(timings))


def add_memory_arguments(parser: argparse.ArgumentParser, *, output_required: bool = True) -> None:
    parser.add_argument("--input", type=Path, required=True, help="本地 memory snapshot（.pickle/.pkl/.json）或 timeline CSV")
    parser.add_argument("--output", type=Path, required=output_required, help="公开 PNG 输出文件（.png）")
    parser.add_argument("--workload", choices=tuple(WORKLOAD_LABELS), required=True, help="图中的固定工作负载标签")
    parser.add_argument("--device", type=nonnegative_int, default=None, help="snapshot/CSV 的 CUDA device；省略时仅允许一个有数据的设备")
    parser.add_argument("--max-points", type=positive_int, default=DEFAULT_MAX_POINTS, help="绘图保留的最大 timeline 点数")
    parser.add_argument("--allow-pickle", action="store_true", help="允许读取可信本地 .pickle/.pkl snapshot；pickle 输入绝不应来自不可信来源")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="从本地 profiling artifact 渲染公开、裁剪、脱敏的 PNG；不复制 raw trace/snapshot。",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    memory = subparsers.add_parser("memory", help="渲染一个 active-allocation memory-history timeline")
    add_memory_arguments(memory)

    timeline = subparsers.add_parser("timeline", help="渲染脱敏 Chrome trace 的 CPU/GPU stage 时间线")
    timeline.add_argument("--input", type=Path, required=True, help="本地 Chrome trace JSON，仅提取白名单阶段标记")
    timeline.add_argument("--output", type=Path, required=True, help="公开 PNG 输出文件（.png）")

    profile = subparsers.add_parser("profile", help="渲染 trace_summary 的 CPU/CUDA stage 时间条形图")
    profile.add_argument("--input", type=Path, required=True, help="轻量 trace_summary CSV")
    profile.add_argument("--output", type=Path, required=True, help="公开 PNG 输出文件（.png）")
    profile.add_argument("--run-name", default=None, help="可选：只使用一个 run；该名称不会写入 PNG")
    profile.add_argument("--aggregation", choices=("mean", "sum", "max"), default="mean", help="多 run 的 stage 汇总方式")

    all_assets = subparsers.add_parser("all", help="一次生成 forward、train_step 和 profile 三张固定 PNG")
    all_assets.add_argument("--forward-snapshot", type=Path, required=True, help="forward-only 的本地 snapshot/JSON/CSV")
    all_assets.add_argument("--train-step-snapshot", type=Path, required=True, help="train_step 的本地 snapshot/JSON/CSV")
    all_assets.add_argument("--trace-summary", type=Path, required=True, help="轻量 trace_summary CSV")
    all_assets.add_argument("--output-dir", type=Path, required=True, help="三张公开 PNG 的输出目录")
    all_assets.add_argument("--device", type=nonnegative_int, default=None, help="snapshot/CSV 的 CUDA device；省略时仅允许一个有数据的设备")
    all_assets.add_argument("--max-points", type=positive_int, default=DEFAULT_MAX_POINTS, help="每张 memory timeline 保留的最大点数")
    all_assets.add_argument("--allow-pickle", action="store_true", help="允许读取可信本地 .pickle/.pkl snapshot")
    all_assets.add_argument("--run-name", default=None, help="可选：只使用一个 profile run；该名称不会写入 PNG")
    all_assets.add_argument("--aggregation", choices=("mean", "sum", "max"), default="mean", help="多 run 的 stage 汇总方式")
    return parser


def run_memory_command(args: argparse.Namespace) -> RenderResult:
    timeline = load_memory_timeline(args.input, args.device, args.allow_pickle)
    return render_memory_timeline(timeline, args.workload, args.output, args.max_points)


def run_profile_command(args: argparse.Namespace) -> RenderResult:
    timings = extract_stage_timings(args.input, args.run_name, args.aggregation)
    return render_stage_bars(timings, args.output, args.aggregation)


def run_timeline_command(args: argparse.Namespace) -> RenderResult:
    lanes = load_chrome_trace_lanes(args.input)
    return render_trace_timeline(lanes, args.output)


def run_all_command(args: argparse.Namespace) -> list[RenderResult]:
    output_dir = args.output_dir
    forward = load_memory_timeline(args.forward_snapshot, args.device, args.allow_pickle)
    train_step = load_memory_timeline(args.train_step_snapshot, args.device, args.allow_pickle)
    timings = extract_stage_timings(args.trace_summary, args.run_name, args.aggregation)
    return [
        render_memory_timeline(forward, "forward", output_dir / "memory_forward_timeline.png", args.max_points),
        render_memory_timeline(train_step, "train_step", output_dir / "memory_train_step_timeline.png", args.max_points),
        render_stage_bars(timings, output_dir / "profile_stage_times.png", args.aggregation),
    ]


def print_result(result: RenderResult) -> None:
    """Report only generic artifact classes; avoid leaking output directories."""

    if result.kind == "memory":
        print(f"已生成公开 memory timeline PNG（绘制 {result.displayed_points} 个采样点）")
    elif result.kind == "trace_timeline":
        print(f"已生成公开 Chrome trace 时间线 PNG（绘制 {result.ranges} 个白名单范围）")
    else:
        print(f"已生成公开 profile stage 时间 PNG（{result.stages} 个阶段）")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "memory":
            results = [run_memory_command(args)]
        elif args.command == "timeline":
            results = [run_timeline_command(args)]
        elif args.command == "profile":
            results = [run_profile_command(args)]
        elif args.command == "all":
            results = run_all_command(args)
        else:  # pragma: no cover - argparse enforces the subcommand choices.
            raise AssertionError(f"未知 command：{args.command}")
    except (FileNotFoundError, OSError, ValueError, RuntimeError) as error:
        print(f"render_assets.py: {public_error_text(error)}", file=sys.stderr)
        return 2
    for result in results:
        print_result(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
