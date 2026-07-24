#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MODEL_CONFIGS,
    autocast_context,
    environment_metadata,
    set_seed,
    synchronize,
)
from profiling.io_utils import (
    artifact_name,
    sanitized_command,
    slugify,
    upsert_csv_rows,
    upsert_json_record,
    utc_timestamp,
)
from profiling.nvtx_ranges import stage_range


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P CUDA memory-history capture")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="xl")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--peaks-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    parser.add_argument("--run-name")
    parser.add_argument("--max-history-entries", type=int, default=1_000_000)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if min(args.batch_size, args.context_length, args.max_history_entries) < 1:
        raise SystemExit("batch size, context length, and history entries must be positive")
    if args.warmup < 1:
        raise SystemExit("memory history must start after at least one warm-up step")
    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("memory history requires --device cuda")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable")
    if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected CUDA device does not support BF16")


def resolved_model_config(args: argparse.Namespace) -> dict[str, int]:
    config = dict(MODEL_CONFIGS[args.model_size])
    for key in ("d_model", "d_ff", "num_layers", "num_heads"):
        value = getattr(args, key)
        if value is not None:
            config[key] = value
    if config["d_model"] % config["num_heads"] != 0:
        raise SystemExit("d_model must be divisible by num_heads")
    return config


def default_run_name(args: argparse.Namespace) -> str:
    return slugify(f"{args.model_size}-bs{args.batch_size}-ctx{args.context_length}-{args.mode}-{args.dtype}-memory-seed{args.seed}")


def memory_metrics(device: torch.device) -> dict[str, int]:
    stats = torch.cuda.memory_stats(device)
    return {
        "active_current_bytes": int(stats.get("active_bytes.all.current", 0)),
        "active_peak_bytes": int(stats.get("active_bytes.all.peak", 0)),
        "allocated_current_bytes": int(torch.cuda.memory_allocated(device)),
        "allocated_peak_bytes": int(torch.cuda.max_memory_allocated(device)),
        "reserved_current_bytes": int(torch.cuda.memory_reserved(device)),
        "reserved_peak_bytes": int(torch.cuda.max_memory_reserved(device)),
    }


def sanitize_frame(frame: dict[str, Any]) -> dict[str, Any]:
    filename = str(frame.get("filename", "<unknown>"))
    normalized = filename.replace("\\", "/")
    marker = "assignment2-systems/"
    if marker in normalized:
        normalized = normalized.split(marker, 1)[1]
    elif normalized.startswith("/") or ":/" in normalized:
        normalized = Path(normalized).name
    return {
        "filename": normalized,
        "line": frame.get("line"),
        "name": frame.get("name"),
    }


def infer_phase(frames: list[dict[str, Any]]) -> str:
    joined = " ".join(f"{frame.get('filename', '')} {frame.get('name', '')}".lower() for frame in frames)
    if any(marker in joined for marker in ("optimizer", "torch/optim", "adamw")):
        return "optimizer"
    if any(marker in joined for marker in ("backward", "autograd", "accumulategrad")):
        return "backward"
    if any(marker in joined for marker in ("cross_entropy", "nll_loss")):
        return "loss"
    if any(
        marker in joined
        for marker in (
            "forward",
            "attention",
            "transformer",
            "embedding",
            "rmsnorm",
        )
    ):
        return "forward"
    return "unknown"


def largest_allocation(snapshot: dict[str, Any]) -> dict[str, Any]:
    traced: list[tuple[int, list[dict[str, Any]], str, str]] = []
    resident: list[tuple[int, list[dict[str, Any]], str, str]] = []

    for device_trace in snapshot.get("device_traces", []) or []:
        for entry in device_trace or []:
            if entry.get("action") != "alloc":
                continue
            traced.append(
                (
                    int(entry.get("size", 0)),
                    entry.get("frames", []) or [],
                    "trace_alloc",
                    "device_trace",
                )
            )

    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            frames = block.get("frames", []) or []
            requested = int(block.get("requested_size", block.get("size", 0)))
            resident.append(
                (
                    requested,
                    frames,
                    str(block.get("state", "unknown")),
                    "resident_block",
                )
            )
            for history in block.get("history", []) or []:
                history_frames = history.get("frames", []) or frames
                real_size = int(history.get("real_size", requested))
                traced.append((real_size, history_frames, "history", "block_history"))

    candidates = [candidate for candidate in traced if candidate[1]]
    if not candidates:
        candidates = traced or [candidate for candidate in resident if candidate[1]] or resident
    if not candidates:
        return {
            "bytes": 0,
            "state": "unavailable",
            "source": "unavailable",
            "phase": "unknown",
            "stack": [],
        }
    size, frames, state, source = max(candidates, key=lambda candidate: candidate[0])
    sanitized = [sanitize_frame(frame) for frame in frames[:12]]
    return {
        "bytes": size,
        "state": state,
        "source": source,
        "phase": infer_phase(sanitized),
        "stack": sanitized,
    }


def execute_step(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer | None,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    instrument: bool,
    stage_state: dict[str, str],
) -> torch.Tensor:
    range_kwargs = {"device": device, "enabled": instrument, "use_nvtx": False}
    if args.mode == "forward":
        stage_state["current"] = "forward"
        with torch.no_grad(), stage_range("forward", **range_kwargs), autocast_context(device, args.dtype):
            return model(tokens)

    assert optimizer is not None
    stage_state["current"] = "zero_grad"
    optimizer.zero_grad(set_to_none=True)
    stage_state["current"] = "forward"
    with stage_range("forward", **range_kwargs), autocast_context(device, args.dtype):
        logits = model(tokens)
    stage_state["current"] = "loss"
    loss = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(), labels.reshape(-1))
    stage_state["current"] = "backward"
    with stage_range("backward", **range_kwargs):
        loss.backward()
    stage_state["current"] = "optimizer"
    with stage_range("optimizer", **range_kwargs):
        optimizer.step()
    stage_state["current"] = "complete"
    return loss.detach()


def append_results(
    *,
    args: argparse.Namespace,
    run_name: str,
    config: dict[str, int],
    environment: dict[str, Any],
    status: str,
    failed_stage: str | None,
    error_type: str | None,
    metrics: dict[str, int],
    allocation: dict[str, Any],
) -> None:
    element_bytes = 4 if args.dtype == "fp32" else 2
    residual_bytes = args.batch_size * args.context_length * config["d_model"] * element_bytes
    row = {
        "run_id": run_name,
        "timestamp_utc": utc_timestamp(),
        "status": status,
        "model_size": args.model_size,
        **config,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "failed_stage": failed_stage,
        **metrics,
        "residual_stream_theoretical_bytes": residual_bytes,
        "residual_stream_element_bytes": element_bytes,
        "largest_allocation_bytes": allocation["bytes"],
        "largest_allocation_phase": allocation["phase"],
        "largest_allocation_state": allocation["state"],
        "snapshot_file": artifact_name(args.snapshot),
    }
    upsert_csv_rows(args.peaks_output, [row])
    upsert_json_record(
        args.metadata_output,
        {
            "run_id": run_name,
            "timestamp_utc": row["timestamp_utc"],
            "status": status,
            "failed_stage": failed_stage,
            "error_type": error_type,
            "command": sanitized_command(),
            "configuration": {
                "model_size": args.model_size,
                **config,
                "vocab_size": args.vocab_size,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "mode": args.mode,
                "dtype": args.dtype,
                "seed": args.seed,
                "warmup": args.warmup,
            },
            "environment": environment,
            "metrics": metrics,
            "residual_stream_theoretical_bytes": residual_bytes,
            "largest_allocation": allocation,
            "artifacts": {
                "snapshot": artifact_name(args.snapshot) if args.snapshot.exists() else None,
                "peaks_csv": artifact_name(args.peaks_output),
            },
        },
    )


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    config = resolved_model_config(args)
    run_name = slugify(args.run_name) if args.run_name else default_run_name(args)
    environment = environment_metadata(device)
    stage_state = {"current": "initialization"}
    history_enabled = False
    snapshot_data: dict[str, Any] = {}
    status = "success"
    failed_stage: str | None = None
    error_type: str | None = None

    try:
        set_seed(args.seed)
        model = BasicsTransformerLM(
            vocab_size=args.vocab_size,
            context_length=args.context_length,
            **config,
        ).to(device)
        model.train()
        optimizer = torch.optim.AdamW(model.parameters()) if args.mode == "train_step" else None
        tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)
        labels = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device=device)

        stage_state["current"] = "warmup"
        for _ in range(args.warmup):
            execute_step(
                args=args,
                device=device,
                model=model,
                optimizer=optimizer,
                tokens=tokens,
                labels=labels,
                instrument=False,
                stage_state=stage_state,
            )
            synchronize(device)

        torch.cuda.reset_peak_memory_stats(device)
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._record_memory_history(max_entries=args.max_history_entries)
        history_enabled = True
        execute_step(
            args=args,
            device=device,
            model=model,
            optimizer=optimizer,
            tokens=tokens,
            labels=labels,
            instrument=True,
            stage_state=stage_state,
        )
        synchronize(device)
        stage_state["current"] = "snapshot"
        snapshot_data = torch.cuda.memory._snapshot()
        torch.cuda.memory._dump_snapshot(str(args.snapshot))
    except torch.cuda.OutOfMemoryError:
        status = "oom"
        failed_stage = stage_state["current"]
        error_type = "OutOfMemoryError"
        try:
            synchronize(device)
        except RuntimeError:
            pass
        if history_enabled:
            try:
                snapshot_data = torch.cuda.memory._snapshot()
                torch.cuda.memory._dump_snapshot(str(args.snapshot))
            except RuntimeError:
                snapshot_data = {}
    except RuntimeError as error:
        status = "runtime_error"
        failed_stage = stage_state["current"]
        error_type = type(error).__name__
    finally:
        if history_enabled:
            try:
                torch.cuda.memory._record_memory_history(enabled=None)
            except RuntimeError:
                pass

    metrics = memory_metrics(device)
    allocation = largest_allocation(snapshot_data)
    append_results(
        args=args,
        run_name=run_name,
        config=config,
        environment=environment,
        status=status,
        failed_stage=failed_stage,
        error_type=error_type,
        metrics=metrics,
        allocation=allocation,
    )
    print(f"{run_name}: status={status} allocated_peak={metrics['allocated_peak_bytes']} reserved_peak={metrics['reserved_peak_bytes']} snapshot={args.snapshot}")
    if status == "oom":
        return 2
    if status != "success":
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
