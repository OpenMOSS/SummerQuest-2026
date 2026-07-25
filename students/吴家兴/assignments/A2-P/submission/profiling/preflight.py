#!/usr/bin/env python3
"""Fail-fast checks for CUDA timing, torch.profiler, and memory history."""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

try:
    from .common import public_environment, require_cuda, write_json
except ImportError:
    from common import public_environment, require_cuda, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    device = require_cuda()

    left = torch.randn((1024, 1024), device=device)
    right = torch.randn((1024, 1024), device=device)
    product = left @ right
    torch.cuda.synchronize()
    cuda_ok = bool(torch.isfinite(product).all().item())

    trace_path = args.output_dir / "preflight_trace.json"
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=False,
        profile_memory=False,
        with_stack=False,
    ) as prof:
        with record_function("preflight/matmul"):
            product = left @ right
            product.sum().item()
    torch.cuda.synchronize()
    prof.export_chrome_trace(str(trace_path))
    device_total = sum(
        float(getattr(event, "device_time_total", 0.0))
        for event in prof.key_averages()
    )
    profiler_ok = trace_path.is_file() and trace_path.stat().st_size > 0

    snapshot_path = args.output_dir / "preflight_snapshot.pickle"
    timeline_path = args.output_dir / "preflight_memory_timeline.html"
    torch.cuda.memory._record_memory_history(
        enabled="all",
        context="all",
        stacks="python",
        max_entries=10_000,
    )
    try:
        temporary = torch.randn((2048, 2048), device=device)
        temporary = temporary.square()
        torch.cuda.synchronize()
        snapshot = torch.cuda.memory._snapshot()
    finally:
        torch.cuda.memory._record_memory_history(enabled=None)
    with snapshot_path.open("wb") as handle:
        pickle.dump(snapshot, handle)
    timeline_path.write_text(
        torch.cuda._memory_viz.trace_plot(snapshot),
        encoding="utf-8",
    )
    memory_history_ok = (
        bool(snapshot.get("segments"))
        and snapshot_path.stat().st_size > 0
        and timeline_path.stat().st_size > 0
    )

    payload = {
        "schema_version": 1,
        "environment": public_environment(),
        "checks": {
            "cuda_matmul": {"ok": cuda_ok},
            "torch_profiler_cpu_cuda": {
                "ok": profiler_ok,
                "reported_device_time_us": round(device_total, 3),
                "trace_file": trace_path.name,
            },
            "memory_history_snapshot": {
                "ok": memory_history_ok,
                "snapshot_file": snapshot_path.name,
                "timeline_file": timeline_path.name,
                "segments": len(snapshot.get("segments", [])),
                "trace_devices": len(snapshot.get("device_traces", [])),
            },
        },
    }
    write_json(args.output_dir / "preflight.json", payload)
    if not all(check["ok"] for check in payload["checks"].values()):
        raise RuntimeError(f"preflight failed: {payload['checks']}")
    print("CUDA, torch.profiler, and memory-history preflights passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
