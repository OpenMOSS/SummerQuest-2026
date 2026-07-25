#!/usr/bin/env python3
"""Isolated PyTorch memory-history runs and the prescribed XL fallback chain."""

from __future__ import annotations

import argparse
import csv
import hashlib
import os
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch
from torch.profiler import record_function

try:
    from .common import (
        autocast_context,
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        run_step,
        sanitize_oom,
        write_json,
    )
    from .config import RunConfig, residual_stream_mib
except ImportError:
    from common import (
        autocast_context,
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        run_step,
        sanitize_oom,
        write_json,
    )
    from config import RunConfig, residual_stream_mib


CSV_FIELDS = (
    "run_id",
    "status",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "warmup_mode",
    "warmup_steps",
    "failure_stage",
    "error_type",
    "attempted_allocation",
    "active_peak_mib",
    "allocated_peak_mib",
    "reserved_peak_mib",
    "largest_trace_allocation_mib",
    "largest_final_active_block_mib",
    "residual_stream_mib",
    "snapshot_file",
    "timeline_file",
    "fallback_from",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _safe_stack(frames: list[dict[str, Any]], limit: int = 6) -> list[str]:
    """Keep only public basenames and function names from allocation stacks."""

    safe: list[str] = []
    for frame in frames:
        filename = Path(str(frame.get("filename", "unknown"))).name
        function = str(frame.get("name", "unknown"))
        item = f"{filename}:{function}"
        if item not in safe:
            safe.append(item)
        if len(safe) >= limit:
            break
    return safe


def summarize_snapshot(snapshot: dict[str, Any]) -> dict[str, Any]:
    allocation_events: list[dict[str, Any]] = []
    for device_trace in snapshot.get("device_traces", []):
        allocation_events.extend(
            event for event in device_trace if event.get("action") == "alloc"
        )
    largest_event = max(
        allocation_events,
        key=lambda event: int(event.get("size", 0)),
        default=None,
    )

    active_blocks: list[dict[str, Any]] = []
    for segment in snapshot.get("segments", []):
        active_blocks.extend(
            block
            for block in segment.get("blocks", [])
            if str(block.get("state", "")).startswith("active")
        )
    largest_block = max(
        active_blocks,
        key=lambda block: int(block.get("requested_size", block.get("size", 0))),
        default=None,
    )
    return {
        "segments": len(snapshot.get("segments", [])),
        "allocation_events": len(allocation_events),
        "largest_trace_allocation_mib": (
            round(int(largest_event["size"]) / 2**20, 3)
            if largest_event
            else None
        ),
        "largest_trace_allocation_stack": (
            _safe_stack(largest_event.get("frames", []))
            if largest_event
            else []
        ),
        "largest_final_active_block_mib": (
            round(
                int(
                    largest_block.get(
                        "requested_size",
                        largest_block.get("size", 0),
                    )
                )
                / 2**20,
                3,
            )
            if largest_block
            else None
        ),
        "largest_final_active_block_stack": (
            _safe_stack(largest_block.get("frames", []))
            if largest_block
            else []
        ),
    }


def _memory_point() -> dict[str, float]:
    return {key: round(value, 3) for key, value in memory_stats_mib().items()}


def measured_memory_step(
    config: RunConfig,
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    optimizer: torch.optim.Optimizer | None,
    points: dict[str, dict[str, float]],
    state: dict[str, str],
) -> float | None:
    """Run stages explicitly so an OOM can be attributed to a stage."""

    from cs336_basics.nn_utils import cross_entropy

    if config.mode == "forward":
        state["stage"] = "forward"
        with record_function("forward"), torch.no_grad(), autocast_context(
            config.dtype
        ):
            logits = model(inputs)
        torch.cuda.synchronize()
        points["after_forward"] = _memory_point()
        del logits
        return None

    if optimizer is None:
        raise ValueError("train_step requires an optimizer")
    with record_function("zero_grad"):
        optimizer.zero_grad(set_to_none=True)

    state["stage"] = "forward"
    with record_function("forward"), autocast_context(config.dtype):
        logits = model(inputs)
        loss_tensor = cross_entropy(logits, targets)
    torch.cuda.synchronize()
    points["after_forward"] = _memory_point()

    state["stage"] = "backward"
    with record_function("backward"):
        loss_tensor.backward()
    torch.cuda.synchronize()
    points["after_backward"] = _memory_point()

    state["stage"] = "optimizer"
    with record_function("optimizer"):
        optimizer.step()
    torch.cuda.synchronize()
    points["after_optimizer"] = _memory_point()
    loss = float(loss_tensor.detach().float().item())
    del logits, loss_tensor
    return loss


def single_run(args: argparse.Namespace) -> dict[str, Any]:
    config = RunConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        dtype=args.dtype,
        warmup_steps=args.warmup,
        measurement_steps=1,
        seed=args.seed,
    )
    config.validate()
    model = build_model(config)
    inputs, targets = build_batch(config)
    optimizer = build_optimizer(model) if config.mode == "train_step" else None

    # Forward-only warm-up initializes CUDA kernels while keeping optimizer
    # state allocation inside the recorded complete train step.
    for _ in range(config.warmup_steps):
        run_step(model, inputs, targets, "forward", config.dtype)
        torch.cuda.synchronize()

    points = {"after_model_and_warmup": _memory_point()}
    torch.cuda.reset_peak_memory_stats()
    snapshot: dict[str, Any] | None = None
    status = "success"
    failure_stage = ""
    error: dict[str, str | None] = {
        "error_type": None,
        "attempted_allocation": None,
    }
    loss: float | None = None
    state = {"stage": "recording_setup"}
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=args.max_entries,
    )
    try:
        loss = measured_memory_step(
            config,
            model,
            inputs,
            targets,
            optimizer,
            points,
            state,
        )
    except torch.OutOfMemoryError as exception:
        status = "oom"
        failure_stage = state["stage"]
        error = sanitize_oom(exception)
    finally:
        try:
            snapshot = torch.cuda.memory._snapshot()
        finally:
            torch.cuda.memory._record_memory_history(enabled=None)

    peak = _memory_point()
    if snapshot is None:
        raise RuntimeError("memory snapshot was not produced")
    summary = summarize_snapshot(snapshot)

    args.raw_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = args.raw_dir / f"{config.run_id}.snapshot.pickle"
    timeline_path = args.raw_dir / f"{config.run_id}.timeline.html"
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)
    timeline_path.write_text(
        torch.cuda._memory_viz.trace_plot(snapshot),
        encoding="utf-8",
    )
    metadata = {
        "run_id": config.run_id,
        "status": status,
        "config": config.as_dict(),
        "warmup": {
            "mode": "forward",
            "steps": config.warmup_steps,
            "completed_before_memory_history": True,
        },
        "failure_stage": failure_stage or None,
        **error,
        "loss": loss,
        "memory": peak,
        "memory_points": points,
        "residual_stream_theoretical_mib": round(
            residual_stream_mib(
                config.model_size,
                config.batch_size,
                config.context_length,
            ),
            6,
        ),
        "snapshot_summary": summary,
        "local_snapshot": {
            "file": snapshot_path.name,
            "bytes": snapshot_path.stat().st_size,
            "sha256": _sha256(snapshot_path),
            "committed": False,
        },
        "local_timeline": {
            "file": timeline_path.name,
            "bytes": timeline_path.stat().st_size,
            "sha256": _sha256(timeline_path),
            "committed": False,
        },
    }
    write_json(args.row_json, metadata)
    print(
        f"{config.run_id}: {status}; "
        f"peak_allocated={peak['allocated_peak_mib']:.1f} MiB",
        flush=True,
    )
    return metadata


def _row(metadata: dict[str, Any], fallback_from: str = "") -> dict[str, Any]:
    config = metadata["config"]
    memory = metadata["memory"]
    snapshot = metadata["snapshot_summary"]
    return {
        "run_id": metadata["run_id"],
        "status": metadata["status"],
        "model_size": config["model_size"],
        "batch_size": config["batch_size"],
        "context_length": config["context_length"],
        "mode": config["mode"],
        "dtype": config["dtype"],
        "warmup_mode": metadata["warmup"]["mode"],
        "warmup_steps": metadata["warmup"]["steps"],
        "failure_stage": metadata["failure_stage"] or "",
        "error_type": metadata["error_type"] or "",
        "attempted_allocation": metadata["attempted_allocation"] or "",
        "active_peak_mib": memory["active_peak_mib"],
        "allocated_peak_mib": memory["allocated_peak_mib"],
        "reserved_peak_mib": memory["reserved_peak_mib"],
        "largest_trace_allocation_mib": (
            snapshot["largest_trace_allocation_mib"] or ""
        ),
        "largest_final_active_block_mib": (
            snapshot["largest_final_active_block_mib"] or ""
        ),
        "residual_stream_mib": metadata["residual_stream_theoretical_mib"],
        "snapshot_file": metadata["local_snapshot"]["file"],
        "timeline_file": metadata["local_timeline"]["file"],
        "fallback_from": fallback_from,
    }


def _run_isolated(
    args: argparse.Namespace,
    model_size: str,
    context_length: int,
    mode: str,
    fallback_from: str = "",
) -> tuple[dict[str, Any], dict[str, Any]]:
    run_id = (
        f"{model_size}-b{args.batch_size}-c{context_length}-{mode}-"
        f"{args.dtype}-w{args.warmup}-n1"
    )
    row_json = args.raw_dir / "metadata" / f"{run_id}.json"
    command = [
        sys.executable,
        "-B",
        str(Path(__file__).resolve()),
        "--single-run",
        "--model-size",
        model_size,
        "--batch-size",
        str(args.batch_size),
        "--context-length",
        str(context_length),
        "--mode",
        mode,
        "--dtype",
        args.dtype,
        "--warmup",
        str(args.warmup),
        "--seed",
        str(args.seed),
        "--max-entries",
        str(args.max_entries),
        "--raw-dir",
        str(args.raw_dir),
        "--row-json",
        str(row_json),
    ]
    environment = os.environ.copy()
    environment["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(command, env=environment, check=False)
    if completed.returncode != 0 or not row_json.is_file():
        raise RuntimeError(
            f"isolated memory run failed before writing metadata: {run_id}"
        )
    import json

    metadata = json.loads(row_json.read_text(encoding="utf-8"))
    return metadata, _row(metadata, fallback_from)


def required_suite(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    metadata_runs: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    required = [
        ("xl", 128, "forward"),
        ("xl", 128, "train_step"),
        ("xl", 2048, "forward"),
        ("xl", 2048, "train_step"),
    ]
    for model_size, context_length, mode in required:
        metadata, row = _run_isolated(
            args,
            model_size,
            context_length,
            mode,
        )
        metadata_runs.append(metadata)
        rows.append(row)

    requested = metadata_runs[-1]
    if requested["status"] == "oom":
        fallback_source = requested["run_id"]
        metadata, row = _run_isolated(
            args,
            "xl",
            1024,
            "train_step",
            fallback_from=fallback_source,
        )
        metadata_runs.append(metadata)
        rows.append(row)
        if metadata["status"] == "oom":
            metadata, row = _run_isolated(
                args,
                "large",
                2048,
                "train_step",
                fallback_from=fallback_source,
            )
            metadata_runs.append(metadata)
            rows.append(row)
    return metadata_runs, rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", choices=("required",), default="required")
    parser.add_argument("--single-run", action="store_true")
    parser.add_argument("--model-size", default="xl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, default=128)
    parser.add_argument(
        "--mode",
        choices=("forward", "train_step"),
        default="forward",
    )
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--max-entries", type=int, default=200_000)
    parser.add_argument("--raw-dir", type=Path, required=True)
    parser.add_argument("--row-json", type=Path)
    parser.add_argument("--summary", type=Path)
    parser.add_argument("--metadata", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.single_run:
        if args.row_json is None:
            raise ValueError("--single-run requires --row-json")
        single_run(args)
        return 0
    if args.summary is None or args.metadata is None:
        raise ValueError("suite mode requires --summary and --metadata")

    metadata_runs, rows = required_suite(args)
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=CSV_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    write_json(
        args.metadata,
        {
            "schema_version": 1,
            "experiment": "memory_profiling",
            "history": {
                "implementation": "torch.cuda.memory._record_memory_history",
                "enabled": "all",
                "context": "all",
                "stacks": "python",
                "max_entries": args.max_entries,
            },
            "isolation": "one fresh process per configuration",
            "fallback_order": [
                "xl/context=2048/batch=1",
                "xl/context=1024/batch=1",
                "large/context=2048/batch=1",
            ],
            "raw_policy": "snapshot pickle and full timeline HTML retained locally",
            "environment": public_environment(),
            "runs": metadata_runs,
        },
    )
    print(f"wrote {len(rows)} memory rows")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
