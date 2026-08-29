"""CUDA memory-history capture for forward and full training steps."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .common import autocast_context, build_model, loss_from_logits, synchronize
from .config import get_model_spec


def _snapshot_largest_allocation(snapshot: dict) -> float:
    """Return the largest active allocation in MiB from a memory snapshot."""
    largest = 0
    for segment in snapshot.get("segments", []):
        for block in segment.get("blocks", []):
            if block.get("state") == "active_allocated":
                largest = max(
                    largest,
                    int(block.get("requested_size", block.get("size", 0))),
                )
    return largest / 2**20


def _write_timeline(
    snapshot: dict, path: Path, initial_active: int, initial_reserved: int
) -> None:
    """Write a compact active/reserved allocation timeline for plotting."""
    traces = snapshot.get("device_traces", [])
    if not traces:
        return
    events = traces[0] if isinstance(traces[0], list) else traces
    active, reserved = initial_active, initial_reserved
    allocations: dict[int, int] = {}
    segments: dict[int, int] = {}
    rows = []
    for event in events:
        action = event.get("action")
        size = int(event.get("size", 0) or 0)
        address = int(event.get("addr", 0) or 0)
        if action == "alloc":
            active += size
            allocations[address] = size
        elif action == "free_completed":
            active = max(0, active - allocations.pop(address, size))
        elif action == "segment_alloc":
            reserved += size
            segments[address] = size
        elif action == "segment_free":
            reserved = max(0, reserved - segments.pop(address, size))
        rows.append(
            {
                "time_us": event.get("time_us", len(rows)),
                "active_mib": active / 2**20,
                "reserved_mib": reserved / 2**20,
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f, fieldnames=("time_us", "active_mib", "reserved_mib")
        )
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="xl")
    parser.add_argument("--context-length", type=int, choices=(128, 2048), default=128)
    parser.add_argument("--mode", choices=("forward", "train_step"), default="forward")
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--output-dir", type=Path, default=Path("results/memory"))
    parser.add_argument("--timeline-output", type=Path, default=None)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    row = {
        "model_size": args.model_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "largest_allocation_mib": "",
        "status": "not_run_no_cuda",
    }
    if torch.cuda.is_available():
        device = torch.device("cuda")
        spec = get_model_spec(args.model_size)
        model = build_model(args.model_size, args.context_length, device)
        tokens = torch.randint(
            0, spec.vocab_size, (1, args.context_length), device=device
        )
        targets = torch.randint_like(tokens, high=spec.vocab_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        with autocast_context(device, args.dtype):
            model(tokens)
        synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        initial_active = torch.cuda.memory_allocated(device)
        initial_reserved = torch.cuda.memory_reserved(device)
        torch.cuda.memory._record_memory_history(max_entries=100000)
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            logits = model(tokens)
        if args.mode == "train_step":
            loss_from_logits(logits, targets).backward()
            optimizer.step()
        synchronize(device)
        snapshot = (
            args.output_dir
            / f"{args.model_size}_c{args.context_length}_{args.mode}_{args.dtype}.pickle"
        )
        torch.cuda.memory._dump_snapshot(str(snapshot))
        snapshot_data = torch.cuda.memory._snapshot()
        torch.cuda.memory._record_memory_history(enabled=None)
        row.update(
            peak_allocated_mib=torch.cuda.max_memory_allocated(device) / 2**20,
            peak_reserved_mib=torch.cuda.max_memory_reserved(device) / 2**20,
            largest_allocation_mib=_snapshot_largest_allocation(snapshot_data),
            status="pass",
        )
        if args.timeline_output is not None:
            _write_timeline(
                snapshot_data, args.timeline_output, initial_active, initial_reserved
            )
    peaks = args.output_dir / "peaks.csv"
    exists = peaks.is_file() and peaks.stat().st_size > 0
    with peaks.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    metadata = {
        "measurement_collected": row["status"] == "pass",
        "status": row["status"],
        "history_started_after_warmup": True,
        "model_size": args.model_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
