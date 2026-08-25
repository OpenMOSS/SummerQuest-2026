#!/usr/bin/env python3
"""Capture one CUDA allocator snapshot and render an active-memory timeline.

The command deliberately handles exactly one experiment case.  Model/data
construction and all warm-up steps happen before allocator history is enabled,
so the trace describes only the requested measured step(s).  ``--dry-run`` is
a CPU-only plumbing check: it runs a tiny model but never fabricates a CUDA
snapshot or an authoritative memory result.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from collections.abc import Callable, Iterator
import contextlib
from dataclasses import dataclass
import gc
import json
import os
from pathlib import Path
import pickle
import random
import re
import time
import traceback
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.optimizer import AdamW
from profiling.nvtx_ranges import BACKWARD, FORWARD, OPTIMIZER, PROFILE_MEASURE, PROFILE_WARMUP, annotated_range


SCHEMA_VERSION = "cs336.a2p.memory-snapshot.v1"
MODEL_CONFIGS: dict[str, dict[str, int]] = {
    "large": {"d_model": 1_280, "d_ff": 5_120, "num_layers": 36, "num_heads": 20},
    "xl": {"d_model": 2_560, "d_ff": 10_240, "num_layers": 32, "num_heads": 32},
}
TINY_CONFIG = {"d_model": 32, "d_ff": 64, "num_layers": 2, "num_heads": 4}
VOCAB_SIZE = 10_000
TINY_VOCAB_SIZE = 256
MODES = ("forward", "train_step")
DTYPES = ("fp32", "bf16")
SAFE_SOURCE_ROOTS = ("profiling", "cs336_basics", "cs336_systems", "cs336-basics")
MEMORY_STAT_KEYS = {
    "active_bytes_current": "active_bytes.all.current",
    "active_bytes_peak": "active_bytes.all.peak",
    "allocated_bytes_current": "allocated_bytes.all.current",
    "allocated_bytes_peak": "allocated_bytes.all.peak",
    "reserved_bytes_current": "reserved_bytes.all.current",
    "reserved_bytes_peak": "reserved_bytes.all.peak",
}


class SnapshotError(RuntimeError):
    """Raised when allocator evidence cannot be captured or interpreted."""


StorageKey = tuple[str, int, int]


@dataclass(frozen=True)
class PhaseBoundary:
    """One fully synchronized measured stage, in the snapshot time domain."""

    step: int
    label: str
    start_time_us: int
    end_time_us: int

    def as_dict(self) -> dict[str, int | str]:
        return {
            "step": self.step,
            "label": self.label,
            "start_time_us": self.start_time_us,
            "end_time_us": self.end_time_us,
            "duration_us": self.end_time_us - self.start_time_us,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Capture one CS336 Transformer CUDA memory snapshot.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), default="xl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=2_048)
    parser.add_argument("--mode", choices=MODES, default="train_step")
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("results/memory_snapshot.json"))
    parser.add_argument("--snapshot-output", type=Path, default=Path("results/memory_snapshot.pickle"))
    parser.add_argument("--timeline-output", type=Path, default=Path("results/active_memory_timeline.png"))
    parser.add_argument(
        "--memory-viz-output",
        type=Path,
        default=None,
        help="optional private HTML from torch.cuda._memory_viz.trace_plot; never publish this full snapshot-derived artifact",
    )
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    parser.add_argument(
        "--saved-tensors-block",
        action="store_true",
        help="also analyze tensors saved for backward by one XL TransformerBlock",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="run a tiny CPU plumbing check; no CUDA snapshot or authoritative result",
    )
    return parser


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.batch_size <= 0 or args.context_length <= 0:
        parser.error("--batch-size and --context-length must be positive")
    if args.warmup < 0 or args.steps <= 0:
        parser.error("--warmup must be non-negative and --steps must be positive")
    if args.max_entries <= 0:
        parser.error("--max-entries must be positive")
    if args.saved_tensors_block and args.model_size != "xl":
        parser.error("--saved-tensors-block is defined for --model-size xl only")
    artifact_paths = [args.output, args.snapshot_output, args.timeline_output]
    if args.memory_viz_output is not None:
        artifact_paths.append(args.memory_viz_output)
    if len({path.absolute() for path in artifact_paths}) != len(artifact_paths):
        parser.error("all artifact output paths must be distinct")
    if args.timeline_output.suffix.lower() != ".png":
        parser.error("--timeline-output must end in .png")
    if args.memory_viz_output is not None and args.memory_viz_output.suffix.lower() not in (".html", ".htm"):
        parser.error("--memory-viz-output must end in .html or .htm")
    if args.memory_viz_output is not None and "private" not in {part.casefold() for part in args.memory_viz_output.absolute().parts[:-1]}:
        parser.error("--memory-viz-output must be located under an explicitly named private directory")
    return args


def _safe_basename(path: Path | str) -> str:
    return Path(str(path).replace("\\", "/")).name


def safe_source_path(path: str | Path | None) -> str:
    """Return a public repo-relative source suffix, or only a basename."""

    if path is None:
        return "<unknown>"
    normalized = str(path).replace("\\", "/")
    parts = tuple(part for part in normalized.split("/") if part and part not in (".", ".."))
    for root in SAFE_SOURCE_ROOTS:
        if root in parts:
            index = parts.index(root)
            return "/".join(parts[index:])
    return parts[-1] if parts else "<unknown>"


_UUID_RE = re.compile(r"\b[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}\b")
_JOB_RE = re.compile(r"\bjob-[0-9a-zA-Z-]+\b")
_UNIX_PATH_RE = re.compile(r"(?<![\w.])/(?:[^\s,;:()\[\]{}]+/)*[^\s,;:()\[\]{}]*")
_WINDOWS_PATH_RE = re.compile(r"\b[A-Za-z]:\\(?:[^\s,;:()\[\]{}]+\\)*[^\s,;:()\[\]{}]*")
_HOST_FIELD_RE = re.compile(r"(?i)\b(host(?:name)?|node)\s*[=:]\s*[^\s,;]+")
_PROCESS_ID_RE = re.compile(r"(?i)\b(process|pid)\s*(?:[=:]\s*|\s+)\d+\b")


def sanitize_exception_message(message: str) -> str:
    """Remove paths and run-specific machine identifiers from an error."""

    sanitized = _UUID_RE.sub("<uuid>", str(message))
    sanitized = _JOB_RE.sub("<job>", sanitized)
    sanitized = _HOST_FIELD_RE.sub(lambda match: f"{match.group(1)}=<host>", sanitized)
    sanitized = _PROCESS_ID_RE.sub(lambda match: f"{match.group(1)}=<id>", sanitized)
    sanitized = _WINDOWS_PATH_RE.sub(lambda match: safe_source_path(match.group(0)), sanitized)
    sanitized = _UNIX_PATH_RE.sub(lambda match: safe_source_path(match.group(0)), sanitized)
    return " ".join(sanitized.split())[:4_000]


def logical_command(args: argparse.Namespace) -> list[str]:
    """Reconstruct a complete, deterministic command without private paths."""

    command = [
        "python",
        "profiling/memory_snapshot.py",
        "--model-size",
        args.model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(args.context_length),
        "--mode",
        args.mode,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--steps",
        str(args.steps),
        "--seed",
        str(args.seed),
        "--output",
        _safe_basename(args.output),
        "--snapshot-output",
        _safe_basename(args.snapshot_output),
        "--timeline-output",
        _safe_basename(args.timeline_output),
        "--max-entries",
        str(args.max_entries),
    ]
    if args.memory_viz_output is not None:
        command.extend(("--memory-viz-output", _safe_basename(args.memory_viz_output)))
    if args.saved_tensors_block:
        command.append("--saved-tensors-block")
    if args.dry_run:
        command.append("--dry-run")
    return command


def requested_configuration(args: argparse.Namespace) -> dict[str, Any]:
    dimensions = MODEL_CONFIGS[args.model_size]
    return {
        "model_size": args.model_size,
        **dimensions,
        "vocab_size": VOCAB_SIZE,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "parameter_dtype": "fp32",
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
    }


def effective_configuration(args: argparse.Namespace) -> dict[str, Any]:
    if not args.dry_run:
        return {**requested_configuration(args), "device": "cuda", "is_reduced": False}
    return {
        "model_size": "tiny-dry-run",
        **TINY_CONFIG,
        "vocab_size": TINY_VOCAB_SIZE,
        "batch_size": min(args.batch_size, 1),
        "context_length": min(args.context_length, 16),
        "mode": args.mode,
        "dtype": args.dtype,
        "parameter_dtype": "fp32",
        "warmup": args.warmup,
        "steps": args.steps,
        "seed": args.seed,
        "device": "cpu",
        "is_reduced": True,
    }


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _autocast(device: torch.device, dtype: str) -> contextlib.AbstractContextManager[Any]:
    if dtype == "fp32":
        return contextlib.nullcontext()
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def _build_model_and_data(
    config: dict[str, Any],
    device: torch.device,
) -> tuple[BasicsTransformerLM, torch.Tensor, torch.Tensor]:
    model = BasicsTransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
    ).to(device=device, dtype=torch.float32)
    model.train(config["mode"] == "train_step")
    input_ids = torch.randint(
        0,
        config["vocab_size"],
        (config["batch_size"], config["context_length"]),
        device=device,
    )
    targets = torch.randint(
        0,
        config["vocab_size"],
        (config["batch_size"], config["context_length"]),
        device=device,
    )
    return model, input_ids, targets


def _run_stage(
    *,
    label: str,
    step: int,
    device: torch.device,
    measured: bool,
    action: Callable[[], Any],
    boundaries: list[PhaseBoundary],
) -> Any:
    _sync(device)
    started_us = time.time_ns() // 1_000
    with annotated_range(label, device=device, record_function=False):
        value = action()
        _sync(device)
    ended_us = time.time_ns() // 1_000
    if measured:
        boundaries.append(PhaseBoundary(step=step, label=label, start_time_us=started_us, end_time_us=ended_us))
    return value


def _execute_step(
    *,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer | None,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype: str,
    device: torch.device,
    step: int,
    measured: bool,
    boundaries: list[PhaseBoundary],
) -> float | None:
    if mode == "forward":

        def forward_only() -> torch.Tensor:
            with torch.inference_mode(), _autocast(device, dtype):
                return model(input_ids)

        logits = _run_stage(
            label=FORWARD,
            step=step,
            device=device,
            measured=measured,
            action=forward_only,
            boundaries=boundaries,
        )
        del logits
        return None

    if optimizer is None:
        raise ValueError("train_step requires an optimizer")
    optimizer.zero_grad(set_to_none=True)

    def train_forward() -> torch.Tensor:
        with _autocast(device, dtype):
            logits = model(input_ids)
            return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))

    loss = _run_stage(
        label=FORWARD,
        step=step,
        device=device,
        measured=measured,
        action=train_forward,
        boundaries=boundaries,
    )
    _run_stage(
        label=BACKWARD,
        step=step,
        device=device,
        measured=measured,
        action=loss.backward,
        boundaries=boundaries,
    )
    _run_stage(
        label=OPTIMIZER,
        step=step,
        device=device,
        measured=measured,
        action=optimizer.step,
        boundaries=boundaries,
    )
    value = float(loss.detach())
    del loss
    return value


def _start_memory_history(max_entries: int) -> None:
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    if recorder is None:
        raise SnapshotError("this PyTorch build does not expose CUDA allocator memory history")
    try:
        recorder(enabled="all", context="all", stacks="all", max_entries=max_entries)
    except TypeError:
        recorder(enabled=True, context="all", stacks="all", max_entries=max_entries)


def _stop_memory_history() -> None:
    recorder = getattr(torch.cuda.memory, "_record_memory_history", None)
    if recorder is not None:
        try:
            recorder(enabled=None)
        except TypeError:
            recorder(enabled=False)


def _dump_memory_snapshot(output: Path) -> None:
    dumper = getattr(torch.cuda.memory, "_dump_snapshot", None)
    if dumper is None:
        raise SnapshotError("this PyTorch build does not expose CUDA allocator snapshot dumping")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        dumper(str(temporary))
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def cuda_memory_counters(device: torch.device) -> dict[str, int | None]:
    counters: dict[str, int | None] = {name: None for name in MEMORY_STAT_KEYS}
    counters["max_memory_allocated"] = None
    if device.type != "cuda" or not torch.cuda.is_available():
        return counters
    stats = torch.cuda.memory_stats(device)
    for output_name, torch_name in MEMORY_STAT_KEYS.items():
        value = stats.get(torch_name)
        counters[output_name] = int(value) if value is not None else None
    counters["max_memory_allocated"] = int(torch.cuda.max_memory_allocated(device))
    return counters


def _record_cuda_memory_counters(result: dict[str, Any], device: torch.device) -> None:
    """Best-effort counter capture that never replaces the original failure."""

    try:
        result["memory"].update(cuda_memory_counters(device))
    except Exception as error:
        result["memory"]["counter_capture_error"] = {
            "type": type(error).__name__,
            "message": sanitize_exception_message(str(error)),
        }


def _safe_stack(frames: Any) -> list[dict[str, int | str]]:
    if not isinstance(frames, list):
        return []
    result: list[dict[str, int | str]] = []
    for frame in frames[-20:]:
        if not isinstance(frame, dict):
            continue
        filename = frame.get("filename", frame.get("file", "<unknown>"))
        function = frame.get("name", frame.get("function", "<unknown>"))
        line = frame.get("line", frame.get("line_number", 0))
        try:
            line_number = max(0, int(line))
        except (TypeError, ValueError):
            line_number = 0
        result.append(
            {
                "file": safe_source_path(str(filename)),
                "line": line_number,
                "function": re.sub(r"[^A-Za-z0-9_<>.]+", "_", str(function))[:160],
            }
        )
    return result


def _trace_events(snapshot: dict[str, Any], device_index: int | None = None) -> tuple[int, list[dict[str, Any]]]:
    traces = snapshot.get("device_traces", [])
    if not isinstance(traces, list) or not traces:
        raise SnapshotError("snapshot has no device_traces")
    if device_index is None or not (0 <= device_index < len(traces)):
        candidates = [(index, trace) for index, trace in enumerate(traces) if isinstance(trace, list)]
        if not candidates:
            raise SnapshotError("snapshot device_traces are malformed")
        device_index, selected = max(candidates, key=lambda item: len(item[1]))
    else:
        selected = traces[device_index]
    if not isinstance(selected, list):
        raise SnapshotError("selected device trace is malformed")
    return device_index, [event for event in selected if isinstance(event, dict)]


def _final_active_bytes(snapshot: dict[str, Any], device_index: int) -> int:
    segments = snapshot.get("segments", [])
    if not isinstance(segments, list):
        return 0
    total = 0
    for segment in segments:
        if not isinstance(segment, dict) or int(segment.get("device", device_index)) != device_index:
            continue
        active_size = segment.get("active_size")
        if isinstance(active_size, (int, float)):
            total += max(0, int(active_size))
            continue
        blocks = segment.get("blocks", [])
        if isinstance(blocks, list):
            total += sum(max(0, int(block.get("size", 0))) for block in blocks if isinstance(block, dict) and str(block.get("state", "")).startswith("active"))
    return total


def parse_memory_snapshot(snapshot: dict[str, Any], device_index: int | None = None) -> dict[str, Any]:
    """Summarize allocator events and reconstruct active bytes over time.

    Allocations that predate history are recovered as a baseline from the
    snapshot's final active bytes and the net ``alloc``/``free_completed``
    delta.  ``free_requested`` is intentionally not treated as a release.
    """

    selected_index, events = _trace_events(snapshot, device_index)
    relevant: list[dict[str, Any]] = []
    maximum: dict[str, Any] | None = None
    for trace_index, event in enumerate(events):
        action = event.get("action")
        size = event.get("size")
        event_time = event.get("time_us")
        if action not in ("alloc", "free_completed") or not isinstance(size, (int, float)):
            continue
        clean = {
            "action": action,
            "size": max(0, int(size)),
            "time_us": int(event_time) if isinstance(event_time, (int, float)) else None,
            "trace_index": trace_index,
            "event_index": len(relevant) + 1,
        }
        relevant.append(clean)
        if action == "alloc" and (maximum is None or clean["size"] > maximum["bytes"]):
            maximum = {"bytes": clean["size"], "stack": _safe_stack(event.get("frames", event.get("frame", [])))}

    final_active = _final_active_bytes(snapshot, selected_index)
    net_delta = sum(event["size"] if event["action"] == "alloc" else -event["size"] for event in relevant)
    raw_baseline = final_active - net_delta
    baseline = max(0, raw_baseline)
    timestamps_available = bool(relevant) and all(event["time_us"] is not None for event in relevant)
    initial_time = relevant[0]["time_us"] if timestamps_available else None
    points = [{"event_index": 0, "time_us": initial_time, "active_bytes": baseline}]
    active = baseline
    underflow_events = 0
    for event in relevant:
        active += event["size"] if event["action"] == "alloc" else -event["size"]
        if active < 0:
            underflow_events += 1
            active = 0
        points.append(
            {
                "event_index": event["event_index"],
                "time_us": event["time_us"] if timestamps_available else None,
                "active_bytes": active,
            }
        )

    return {
        "device_trace_index": selected_index,
        "trace_event_count": len(events),
        "active_memory_event_count": len(relevant),
        "final_active_bytes_from_segments": final_active,
        "history_baseline_active_bytes": baseline,
        "baseline_was_clamped": raw_baseline < 0,
        "timeline_underflow_event_count": underflow_events,
        "timeline_x_axis": "snapshot_time_us" if timestamps_available else "allocator_event_index",
        "timeline_points": points,
        "maximum_single_allocation_bytes": maximum["bytes"] if maximum else None,
        "maximum_single_allocation_stack": maximum["stack"] if maximum else [],
        "timeline_policy": "baseline plus alloc and free_completed events; free_requested is not a completed release",
    }


def load_snapshot(path: Path) -> dict[str, Any]:
    """Load the allocator snapshot created by this process."""

    with path.open("rb") as handle:
        snapshot = pickle.load(handle)  # noqa: S301 - this file is produced by this process
    if not isinstance(snapshot, dict):
        raise SnapshotError("allocator snapshot root is not a dictionary")
    return snapshot


def load_and_parse_snapshot(path: Path, device_index: int | None = None) -> dict[str, Any]:
    return parse_memory_snapshot(load_snapshot(path), device_index)


def render_official_memory_viz(snapshot: dict[str, Any], output: Path, *, device_index: int) -> None:
    """Atomically write PyTorch's full private Active Memory Timeline HTML."""

    memory_viz = getattr(torch.cuda, "_memory_viz", None)
    trace_plot = getattr(memory_viz, "trace_plot", None)
    if not callable(trace_plot):
        raise SnapshotError("this PyTorch build does not expose torch.cuda._memory_viz.trace_plot")
    html = trace_plot(snapshot, device=torch.device("cuda", device_index))
    if not isinstance(html, str) or not html.strip():
        raise SnapshotError("torch.cuda._memory_viz.trace_plot returned empty HTML")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    try:
        with temporary.open("x", encoding="utf-8") as handle:
            handle.write(html)
        os.replace(temporary, output)
    finally:
        if temporary.exists():
            temporary.unlink()


def render_active_memory_timeline(
    timeline: dict[str, Any],
    output: Path,
    *,
    configuration: dict[str, Any],
    phase_boundaries: list[dict[str, Any]] | None = None,
) -> None:
    """Render a genuine active-memory timeline without environment metadata."""

    points = timeline.get("timeline_points", [])
    if not isinstance(points, list) or not points:
        raise SnapshotError("cannot render an empty active-memory timeline")
    import matplotlib

    matplotlib.use("Agg", force=True)
    from matplotlib import pyplot as plt

    values_mib = [int(point["active_bytes"]) / 2**20 for point in points]
    use_timestamps = timeline.get("timeline_x_axis") == "snapshot_time_us"
    if use_timestamps:
        times = [int(point["time_us"]) for point in points]
        origin = min(times)
        x_values = [(value - origin) / 1_000.0 for value in times]
        x_label = "elapsed snapshot trace time (ms)"
    else:
        times = []
        origin = 0
        x_values = [int(point["event_index"]) for point in points]
        x_label = "allocator alloc/free_completed event index"

    figure, axis = plt.subplots(figsize=(11, 5.5))
    axis.step(x_values, values_mib, where="post", linewidth=1.4, color="#3465a4", label="active allocator bytes")
    axis.fill_between(x_values, values_mib, step="post", alpha=0.16, color="#3465a4")
    labels_seen: set[str] = set()
    palette = {FORWARD: "#2ca02c", BACKWARD: "#d62728", OPTIMIZER: "#9467bd"}
    if use_timestamps:
        trace_min, trace_max = min(times), max(times)
        tolerance = max(1_000, trace_max - trace_min)
        for boundary in phase_boundaries or []:
            label = str(boundary.get("label", "phase"))
            start = int(boundary.get("start_time_us", 0))
            if not (trace_min - tolerance <= start <= trace_max + tolerance):
                continue
            legend_label = label if label not in labels_seen else None
            labels_seen.add(label)
            axis.axvline(
                (start - origin) / 1_000.0,
                color=palette.get(label, "#777777"),
                linestyle="--",
                linewidth=0.9,
                alpha=0.85,
                label=legend_label,
            )

    axis.set_xlabel(x_label)
    axis.set_ylabel("active memory (MiB)")
    axis.set_title(f"Active Memory Timeline — {configuration['model_size']} / context {configuration['context_length']} / {configuration['mode']} / {configuration['dtype']}")
    axis.grid(True, alpha=0.2)
    if labels_seen:
        axis.legend(loc="best")
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.png")
    try:
        figure.savefig(temporary, dpi=150, metadata={"Software": "CS336 A2-P memory snapshot"})
        os.replace(temporary, output)
    finally:
        plt.close(figure)
        if temporary.exists():
            temporary.unlink()


def theoretical_block_parameter_count(d_model: int, d_ff: int) -> int:
    return 4 * d_model**2 + 3 * d_model * d_ff + 2 * d_model


def theoretical_fp32_block_gradient_bytes(d_model: int, d_ff: int) -> int:
    return theoretical_block_parameter_count(d_model, d_ff) * torch.empty((), dtype=torch.float32).element_size()


def _storage_key(tensor: torch.Tensor) -> StorageKey:
    storage = tensor.untyped_storage()
    return str(storage.device), int(storage.data_ptr()), int(storage.nbytes())


def _saved_tensor_source(active_modules: tuple[str, ...]) -> dict[str, Any]:
    stack = traceback.extract_stack(limit=60)[:-2]
    public_frames = [frame for frame in stack if any(root in Path(frame.filename).parts for root in SAFE_SOURCE_ROOTS)]
    source = public_frames[-1] if public_frames else stack[-1]
    source_file = safe_source_path(source.filename)
    module_path = active_modules[-1] if active_modules else "functional"
    function = re.sub(r"[^A-Za-z0-9_<>.]+", "_", source.name)[:160]
    operation = f"{module_path} | {source_file}:{source.lineno}:{function}"
    return {
        "operation": operation,
        "module_path": module_path,
        "source_file": source_file,
        "source_line": max(0, int(source.lineno)),
        "source_function": function,
    }


class SavedTensorRecorder:
    """Deduplicate non-parameter saved storages and record backward retrievals."""

    def __init__(self, parameter_storage_keys: set[StorageKey] | None = None) -> None:
        self.parameter_storage_keys = parameter_storage_keys or set()
        self.active_modules: list[str] = []
        self._started_ns = time.perf_counter_ns()
        self._storage_ids: dict[StorageKey, str] = {}
        self._storage_rows: dict[StorageKey, dict[str, Any]] = {}
        self._logical_by_operation: defaultdict[str, int] = defaultdict(int)
        self._event_count_by_operation: defaultdict[str, int] = defaultdict(int)
        self._parameter_event_count = 0
        self._saved_event_count = 0
        self.release_events: list[dict[str, Any]] = []

    def _storage_id(self, key: StorageKey) -> str:
        if key not in self._storage_ids:
            self._storage_ids[key] = f"storage-{len(self._storage_ids) + 1:04d}"
        return self._storage_ids[key]

    def pack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"saved_tensors_hooks pack expected Tensor, got {type(tensor).__name__}")
        self._saved_event_count += 1
        key = _storage_key(tensor)
        if key in self.parameter_storage_keys:
            self._parameter_event_count += 1
            return tensor
        attribution = _saved_tensor_source(tuple(self.active_modules))
        operation = str(attribution["operation"])
        self._logical_by_operation[operation] += tensor.numel() * tensor.element_size()
        self._event_count_by_operation[operation] += 1
        if key not in self._storage_rows:
            self._storage_rows[key] = {
                "storage_id": self._storage_id(key),
                "bytes": key[2],
                **attribution,
            }
        return tensor

    def unpack(self, tensor: torch.Tensor) -> torch.Tensor:
        if not isinstance(tensor, torch.Tensor):
            raise TypeError(f"saved_tensors_hooks unpack expected Tensor, got {type(tensor).__name__}")
        key = _storage_key(tensor)
        row = self._storage_rows.get(key)
        if row is not None:
            self.release_events.append(
                {
                    "sequence": len(self.release_events) + 1,
                    "storage_id": row["storage_id"],
                    "bytes": row["bytes"],
                    "retrieved_at_us_since_analysis_start": (time.perf_counter_ns() - self._started_ns) // 1_000,
                    "semantics": "saved tensor retrieved by backward; this is a release opportunity, not proof of allocator free",
                }
            )
        return tensor

    def summarize(self) -> dict[str, Any]:
        operation_rows: dict[str, dict[str, Any]] = {}
        for storage in self._storage_rows.values():
            operation = str(storage["operation"])
            row = operation_rows.setdefault(
                operation,
                {
                    "operation": operation,
                    "module_path": storage["module_path"],
                    "source_file": storage["source_file"],
                    "source_line": storage["source_line"],
                    "source_function": storage["source_function"],
                    "unique_saved_bytes": 0,
                    "unique_storage_count": 0,
                },
            )
            row["unique_saved_bytes"] += int(storage["bytes"])
            row["unique_storage_count"] += 1
        for operation, row in operation_rows.items():
            row["logical_saved_bytes"] = self._logical_by_operation[operation]
            row["saved_event_count"] = self._event_count_by_operation[operation]
        operations = sorted(
            operation_rows.values(),
            key=lambda row: (-int(row["unique_saved_bytes"]), -int(row["logical_saved_bytes"]), str(row["operation"])),
        )
        for rank, row in enumerate(operations, start=1):
            row["rank"] = rank
        unique_bytes = sum(int(row["bytes"]) for row in self._storage_rows.values())
        return {
            "saved_tensor_event_count": self._saved_event_count,
            "parameter_saved_event_count_excluded": self._parameter_event_count,
            "unique_saved_storage_count": len(self._storage_rows),
            "unique_saved_bytes": unique_bytes,
            "top_5_operations": operations[:5],
            "operation_count": len(operations),
            "release_event_count": len(self.release_events),
            "release_events": list(self.release_events),
            "accounting_policy": {
                "unique_saved_bytes": "full untyped-storage bytes, deduplicated by storage; persistent parameter storages excluded",
                "release_events": "saved-tensor backward retrievals, which are release opportunities rather than allocator-free observations",
            },
        }


@contextlib.contextmanager
def _track_active_modules(block: torch.nn.Module, recorder: SavedTensorRecorder) -> Iterator[None]:
    handles: list[torch.utils.hooks.RemovableHandle] = []

    def pre_hook(label: str) -> Callable[[torch.nn.Module, tuple[Any, ...]], None]:
        def enter(_module: torch.nn.Module, _inputs: tuple[Any, ...]) -> None:
            recorder.active_modules.append(label)

        return enter

    def post_hook(label: str) -> Callable[[torch.nn.Module, tuple[Any, ...], Any], None]:
        def leave(_module: torch.nn.Module, _inputs: tuple[Any, ...], _output: Any) -> None:
            if recorder.active_modules and recorder.active_modules[-1] == label:
                recorder.active_modules.pop()

        return leave

    for name, module in block.named_modules():
        label = name or "TransformerBlock"
        handles.append(module.register_forward_pre_hook(pre_hook(label)))
        handles.append(module.register_forward_hook(post_hook(label), always_call=True))
    try:
        yield
    finally:
        for handle in handles:
            handle.remove()
        recorder.active_modules.clear()


def analyze_saved_tensors_block(
    model: BasicsTransformerLM,
    *,
    configuration: dict[str, Any],
    device: torch.device,
) -> dict[str, Any]:
    """Independently profile one block after allocator-history capture ends."""

    block = model.layers[0]
    block.zero_grad(set_to_none=True)
    parameter_keys = {_storage_key(parameter) for parameter in block.parameters()}
    recorder = SavedTensorRecorder(parameter_keys)
    input_tensor = torch.randn(
        configuration["batch_size"],
        configuration["context_length"],
        configuration["d_model"],
        device=device,
        dtype=torch.float32,
        requires_grad=True,
    )
    with torch.autograd.graph.saved_tensors_hooks(recorder.pack, recorder.unpack), _track_active_modules(block, recorder):
        output = block(input_tensor)
        loss = output.sum()
        loss.backward()
        _sync(device)
    summary = recorder.summarize()
    theoretical_bytes = theoretical_fp32_block_gradient_bytes(configuration["d_model"], configuration["d_ff"])
    actual_gradient_bytes = sum(parameter.grad.numel() * parameter.grad.element_size() for parameter in block.parameters() if parameter.grad is not None)
    result = {
        "authoritative": configuration.get("model_size") == "xl" and device.type == "cuda",
        "scope": "one TransformerBlock",
        "configuration": {
            "d_model": configuration["d_model"],
            "d_ff": configuration["d_ff"],
            "num_heads": configuration["num_heads"],
            "batch_size": configuration["batch_size"],
            "context_length": configuration["context_length"],
            "parameter_dtype": "fp32",
        },
        "saved_tensors": summary,
        "parameter_gradients": {
            "formula": "(4*D^2 + 3*D*D_ff + 2*D) * 4 bytes",
            "theoretical_fp32_block_gradient_bytes": theoretical_bytes,
            "actual_block_gradient_bytes": actual_gradient_bytes,
            "matches_theory": actual_gradient_bytes == theoretical_bytes,
        },
    }
    del loss, output, input_tensor
    block.zero_grad(set_to_none=True)
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
        _sync(device)
    return result


def _is_oom(error: BaseException) -> bool:
    oom_types: list[type[BaseException]] = [torch.OutOfMemoryError]
    cuda_oom = getattr(torch.cuda, "OutOfMemoryError", None)
    if isinstance(cuda_oom, type) and issubclass(cuda_oom, BaseException):
        oom_types.append(cuda_oom)
    return isinstance(error, tuple(oom_types)) or "out of memory" in str(error).lower()


def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
            handle.write("\n")
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _record_snapshot_processing_error(result: dict[str, Any], error: BaseException) -> None:
    """Record derived-artifact failure without masking an earlier measured-step error."""

    processing_error = {
        "type": type(error).__name__,
        "message": sanitize_exception_message(str(error)),
    }
    result["snapshot"]["processing_error"] = processing_error
    if result["status"] == "ok":
        result["status"] = "oom" if _is_oom(error) else "error"
        result["exception"] = processing_error


def _base_result(args: argparse.Namespace, config: dict[str, Any]) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "error",
        "authoritative": not args.dry_run,
        "dry_run": args.dry_run,
        "configuration": requested_configuration(args),
        "effective_configuration": config,
        "logical_command": logical_command(args),
        "artifacts": {
            "output_basename": _safe_basename(args.output),
            "snapshot_basename": _safe_basename(args.snapshot_output),
            "snapshot_generated": False,
            "timeline_basename": _safe_basename(args.timeline_output),
            "timeline_generated": False,
            "memory_viz_basename": _safe_basename(args.memory_viz_output) if args.memory_viz_output is not None else None,
            "memory_viz_generated": False,
            "timeline_derived_from_same_snapshot": False,
        },
        "measurement": {
            "warmup_completed": 0,
            "steps_completed": 0,
            "phase_boundaries": [],
            "measured_elapsed_ms": None,
            "last_loss": None,
            "history_started_after_warmup": False,
        },
        "memory": {
            **{name: None for name in MEMORY_STAT_KEYS},
            "max_memory_allocated": None,
            "maximum_single_allocation_bytes": None,
            "maximum_single_allocation_stack": [],
        },
        "snapshot": {
            "generated": False,
            "basename": _safe_basename(args.snapshot_output),
            "timeline_policy": None,
            "summary": None,
        },
        "saved_tensors_block": None,
        "exception": None,
        "limitations": (
            ["CPU tiny dry-run only; CUDA allocator history, snapshot, timeline, and memory numbers are intentionally absent and non-authoritative"] if args.dry_run else []
        ),
        "runtime": {
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu_model": None,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    config = effective_configuration(args)
    result = _base_result(args, config)
    device = torch.device(config["device"])
    model: BasicsTransformerLM | None = None
    optimizer: torch.optim.Optimizer | None = None
    input_ids: torch.Tensor | None = None
    targets: torch.Tensor | None = None
    history_active = False
    snapshot_generated = False
    boundaries: list[PhaseBoundary] = []
    raised: Exception | None = None

    try:
        if not args.dry_run and not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable; use --dry-run for a non-authoritative CPU plumbing check")
        _seed_everything(args.seed)
        if device.type == "cuda":
            result["runtime"]["gpu_model"] = torch.cuda.get_device_name(device)
        model, input_ids, targets = _build_model_and_data(config, device)
        optimizer = AdamW(model.parameters(), lr=1e-3) if args.mode == "train_step" else None

        with annotated_range(PROFILE_WARMUP, device=device, record_function=False):
            for step in range(args.warmup):
                _execute_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                    mode=args.mode,
                    dtype=args.dtype,
                    device=device,
                    step=step,
                    measured=False,
                    boundaries=boundaries,
                )
                result["measurement"]["warmup_completed"] += 1

        if device.type == "cuda":
            _sync(device)
            torch.cuda.reset_peak_memory_stats(device)
            _start_memory_history(args.max_entries)
            history_active = True
            result["measurement"]["history_started_after_warmup"] = True

        _sync(device)
        measured_started = time.perf_counter()
        last_loss: float | None = None
        with annotated_range(PROFILE_MEASURE, device=device, record_function=False):
            for step in range(args.steps):
                last_loss = _execute_step(
                    model=model,
                    optimizer=optimizer,
                    input_ids=input_ids,
                    targets=targets,
                    mode=args.mode,
                    dtype=args.dtype,
                    device=device,
                    step=step,
                    measured=True,
                    boundaries=boundaries,
                )
                result["measurement"]["steps_completed"] += 1
        _sync(device)
        result["measurement"]["measured_elapsed_ms"] = (time.perf_counter() - measured_started) * 1_000.0
        result["measurement"]["last_loss"] = last_loss
        _record_cuda_memory_counters(result, device)

        if device.type == "cuda":
            _dump_memory_snapshot(args.snapshot_output)
            snapshot_generated = True
        result["status"] = "ok"
    except Exception as error:  # preserve a result artifact for both expected OOM and ordinary failures
        raised = error
        result["status"] = "oom" if _is_oom(error) else "error"
        result["exception"] = {
            "type": type(error).__name__,
            "message": sanitize_exception_message(str(error)),
        }
        _record_cuda_memory_counters(result, device)
        if history_active and not snapshot_generated:
            try:
                _dump_memory_snapshot(args.snapshot_output)
                snapshot_generated = True
            except Exception as snapshot_error:
                result["snapshot"]["capture_error"] = {
                    "type": type(snapshot_error).__name__,
                    "message": sanitize_exception_message(str(snapshot_error)),
                }
    finally:
        if history_active:
            try:
                _stop_memory_history()
            except Exception as stop_error:
                result["snapshot"]["history_stop_error"] = {
                    "type": type(stop_error).__name__,
                    "message": sanitize_exception_message(str(stop_error)),
                }
        result["measurement"]["phase_boundaries"] = [boundary.as_dict() for boundary in boundaries]

    if snapshot_generated:
        result["artifacts"]["snapshot_generated"] = True
        result["snapshot"]["generated"] = True
        try:
            device_index = torch.cuda.current_device() if device.type == "cuda" else None
            snapshot = load_snapshot(args.snapshot_output)
            summary = parse_memory_snapshot(snapshot, device_index)
            result["snapshot"]["summary"] = summary
            result["snapshot"]["timeline_policy"] = summary["timeline_policy"]
            result["memory"]["maximum_single_allocation_bytes"] = summary["maximum_single_allocation_bytes"]
            result["memory"]["maximum_single_allocation_stack"] = summary["maximum_single_allocation_stack"]
            if args.memory_viz_output is not None:
                if device_index is None:
                    raise SnapshotError("official memory visualization requires a CUDA device index")
                render_official_memory_viz(snapshot, args.memory_viz_output, device_index=device_index)
                result["artifacts"]["memory_viz_generated"] = True
            render_active_memory_timeline(
                summary,
                args.timeline_output,
                configuration=requested_configuration(args),
                phase_boundaries=result["measurement"]["phase_boundaries"],
            )
            result["artifacts"]["timeline_generated"] = True
            result["artifacts"]["timeline_derived_from_same_snapshot"] = True
        except Exception as artifact_error:
            _record_snapshot_processing_error(result, artifact_error)

    if args.saved_tensors_block and model is not None and raised is None:
        try:
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
                optimizer = None
            model.zero_grad(set_to_none=True)
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()
                _sync(device)
            result["saved_tensors_block"] = analyze_saved_tensors_block(model, configuration=config, device=device)
        except Exception as saved_error:
            result["saved_tensors_block"] = {
                "status": "error",
                "exception": {
                    "type": type(saved_error).__name__,
                    "message": sanitize_exception_message(str(saved_error)),
                },
            }
            if result["status"] == "ok":
                result["status"] = "oom" if _is_oom(saved_error) else "error"
                result["exception"] = result["saved_tensors_block"]["exception"]

    del optimizer, targets, input_ids, model
    gc.collect()
    if device.type == "cuda":
        try:
            torch.cuda.empty_cache()
        except Exception as cleanup_error:
            result["cleanup_error"] = {
                "type": type(cleanup_error).__name__,
                "message": sanitize_exception_message(str(cleanup_error)),
            }
    return result


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    result = run(args)
    _atomic_write_json(args.output, result)
    return 0 if result["status"] in ("ok", "oom") else 1


if __name__ == "__main__":
    raise SystemExit(main())
