from __future__ import annotations

import argparse
import contextlib
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

import torch

from .benchmark import build_parser as benchmark_parser
from .benchmark import build_workload, execute_steps, make_step, synchronize
from .common import command_string, cuda_memory_metrics, local_artifact_name, model_config_dict, software_metadata, utc_now, write_json
from .nvtx_ranges import annotated_attention


@contextlib.contextmanager
def block_memory_observer(model: torch.nn.Module, device: torch.device):
    """Measure saved tensors and backward memory changes for each TransformerBlock."""
    from cs336_basics.model import TransformerBlock

    active_blocks: list[str] = []
    saved_bytes: dict[str, int] = defaultdict(int)
    backward_events: list[dict[str, Any]] = []
    handles = []

    def forward_pre(name: str):
        def hook(_module, _inputs):
            active_blocks.append(name)

        return hook

    def forward_post(name: str):
        def hook(_module, _inputs, _output):
            if not active_blocks or active_blocks[-1] != name:
                raise RuntimeError(f"TransformerBlock observer stack mismatch at {name}")
            active_blocks.pop()

        return hook

    def backward_pre(name: str):
        def hook(_module, _grad_output):
            backward_events.append(
                {
                    "block": name,
                    "event": "backward_start",
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "active_bytes": int(torch.cuda.memory_stats(device).get("active_bytes.all.current", 0)),
                }
            )

        return hook

    def backward_post(name: str):
        def hook(_module, _grad_input, _grad_output):
            backward_events.append(
                {
                    "block": name,
                    "event": "backward_end",
                    "allocated_bytes": int(torch.cuda.memory_allocated(device)),
                    "active_bytes": int(torch.cuda.memory_stats(device).get("active_bytes.all.current", 0)),
                }
            )

        return hook

    for name, module in model.named_modules():
        if isinstance(module, TransformerBlock):
            handles.extend(
                [
                    module.register_forward_pre_hook(forward_pre(name)),
                    module.register_forward_hook(forward_post(name)),
                    module.register_full_backward_pre_hook(backward_pre(name)),
                    module.register_full_backward_hook(backward_post(name)),
                ]
            )

    def pack(tensor: torch.Tensor):
        if active_blocks and not isinstance(tensor, torch.nn.Parameter):
            saved_bytes[active_blocks[-1]] += tensor.numel() * tensor.element_size()
        return tensor

    def unpack(tensor: torch.Tensor):
        return tensor

    try:
        with torch.autograd.graph.saved_tensors_hooks(pack, unpack):
            yield saved_bytes, backward_events
    finally:
        for handle in handles:
            handle.remove()


def build_parser() -> argparse.ArgumentParser:
    parser = benchmark_parser()
    parser.description = "Capture an A2-P PyTorch CUDA memory snapshot"
    parser.set_defaults(model_size="xl", batch_size=1, warmup=1, steps=1)
    parser.add_argument("--snapshot-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--max-entries", type=int, default=1_000_000)
    return parser


def sanitize_error(exc: BaseException) -> str:
    text = str(exc).splitlines()[0]
    allocation = re.search(r"Tried to allocate ([0-9.]+ [KMG]iB)", text)
    return f"CUDA out of memory while requesting {allocation.group(1)}" if allocation else type(exc).__name__


def largest_active_allocation(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    largest: dict[str, Any] | None = None
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") != "active_allocated":
                continue
            candidate: dict[str, Any] = {
                "size_bytes": int(block.get("size", 0)),
                "requested_size_bytes": int(block.get("requested_size", block.get("size", 0))),
            }
            history = block.get("history") or []
            if history:
                frames = history[0].get("frames") or []
                candidate["frames"] = [
                    {"filename": Path(frame.get("filename", "")).name, "line": frame.get("line"), "name": frame.get("name")}
                    for frame in frames[:8]
                ]
            if largest is None or candidate["size_bytes"] > largest["size_bytes"]:
                largest = candidate
    return largest


def largest_recorded_allocation(snapshot: dict[str, Any]) -> dict[str, Any] | None:
    largest: dict[str, Any] | None = None
    for device_trace in snapshot.get("device_traces", []):
        for event in device_trace:
            if event.get("action") != "alloc":
                continue
            candidate: dict[str, Any] = {
                "size_bytes": int(event.get("size", 0)),
                "time_us": int(event.get("time_us", 0)),
                "frames": [],
            }
            for frame in event.get("frames") or []:
                filename = Path(frame.get("filename", "")).name
                if filename in {"model.py", "benchmark.py", "nvtx_ranges.py", "Functions.cpp", "Linear.cpp"}:
                    candidate["frames"].append({"filename": filename, "line": frame.get("line"), "name": frame.get("name")})
                if len(candidate["frames"]) == 6:
                    break
            if largest is None or candidate["size_bytes"] > largest["size_bytes"]:
                largest = candidate
    return largest


def active_memory_timeline(snapshot: dict[str, Any], final_active_bytes: int) -> list[dict[str, int]]:
    """Reconstruct active allocated bytes from allocator history events."""
    events: list[tuple[int, int]] = []
    for device_trace in snapshot.get("device_traces", []):
        for event in device_trace:
            action = event.get("action")
            size = int(event.get("size", 0))
            timestamp = int(event.get("time_us", len(events)))
            if action == "alloc":
                events.append((timestamp, size))
            elif action == "free_requested":
                events.append((timestamp, -size))

    # History begins after warm-up, so model/optimizer allocations predate the first
    # event. Recover that baseline from the final active total and recorded deltas.
    active = max(0, final_active_bytes - sum(delta for _, delta in events))
    timeline = []
    for timestamp, delta in sorted(events, key=lambda item: item[0]):
        active = max(0, active + delta)
        timeline.append({"time_us": timestamp, "active_bytes": active})
    # Downsample while preserving endpoints so public metadata remains small.
    if len(timeline) > 500:
        stride = max(1, math.ceil(len(timeline) / 499))
        timeline = timeline[::stride]
    return timeline


def main() -> None:
    args = build_parser().parse_args()
    model = optimizer = tokens = targets = device = step = None
    summary: dict[str, Any] = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "command": command_string(),
        "summary_file": local_artifact_name(args.summary_output),
        "config": {
            "model_size": args.model_size,
            **model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "mode": args.mode,
            "warmup": args.warmup,
            "dtype": args.dtype,
            "seed": args.seed,
        },
        "snapshot_file": local_artifact_name(args.snapshot_output),
    }
    history_started = False
    try:
        model, optimizer, tokens, targets, device = build_workload(args)
        if device.type != "cuda":
            raise RuntimeError("Memory profiling requires CUDA")
        step = make_step(args, model, optimizer, tokens, targets, device)
        summary["software"] = software_metadata(device)
        with annotated_attention():
            try:
                execute_steps(step, args.warmup, device, "profile/warmup")
            except (torch.OutOfMemoryError, RuntimeError) as exc:
                if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                    setattr(exc, "a2p_failure_stage", "warmup")
                raise
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            torch.cuda.memory._record_memory_history(max_entries=args.max_entries)
            history_started = True
            with block_memory_observer(model, device) as (saved_bytes, backward_events):
                try:
                    execute_steps(step, 1, device, "profile/measure")
                except (torch.OutOfMemoryError, RuntimeError) as exc:
                    if isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower():
                        setattr(exc, "a2p_failure_stage", "measure")
                    raise
            synchronize(device)
            snapshot = torch.cuda.memory._snapshot()
            args.snapshot_output.parent.mkdir(parents=True, exist_ok=True)
            torch.cuda.memory._dump_snapshot(str(args.snapshot_output))
        memory = cuda_memory_metrics(device)
        summary.update(
            {
                "status": "ok",
                "memory": memory,
                "largest_active_allocation": largest_active_allocation(snapshot),
                "largest_recorded_allocation": largest_recorded_allocation(snapshot),
                "active_memory_timeline": active_memory_timeline(snapshot, int(memory["active_bytes"] or 0)),
                "transformer_blocks": {
                    "saved_tensor_bytes": dict(saved_bytes),
                    "backward_memory_events": backward_events,
                },
            }
        )
    except (torch.OutOfMemoryError, RuntimeError) as exc:
        is_oom = isinstance(exc, torch.OutOfMemoryError) or "out of memory" in str(exc).lower()
        summary.update(
            {
                "status": "oom" if is_oom else "error",
                "error_type": type(exc).__name__,
                "error": sanitize_error(exc),
                "failure_stage": getattr(exc, "a2p_failure_stage", "setup"),
            }
        )
        if isinstance(device, torch.device) and device.type == "cuda":
            summary["memory"] = cuda_memory_metrics(device)
        if not is_oom:
            write_json(args.summary_output, summary)
            raise
    finally:
        if history_started:
            torch.cuda.memory._record_memory_history(enabled=None)
    write_json(args.summary_output, summary)
    print(f"wrote {args.summary_output} ({summary['status']})")


if __name__ == "__main__":
    main()
