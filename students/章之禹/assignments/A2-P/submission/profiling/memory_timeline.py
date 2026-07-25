"""Render a small, static Active Memory Timeline from a CUDA snapshot.

The assignment's interactive memory_viz page is useful for exploration, but a
public submission should contain a cropped, dependency-light image rather than
the full HTML viewer or the pickle snapshot.  This helper reconstructs the
active allocation curve from the snapshot event stream and writes an SVG.
"""

from __future__ import annotations

import argparse
import pickle
from pathlib import Path
from typing import Any


def _active_series(snapshot: dict[str, Any], device: int = 0) -> tuple[list[int], list[int]]:
    traces = snapshot.get("device_traces", [])
    if not traces:
        return [], []
    if device < 0 or device >= len(traces):
        raise ValueError(f"snapshot has no device trace {device}")

    # Memory history starts after warm-up, so parameters and other tensors that
    # were already live do not have an ``alloc`` event in this trace.  Recover
    # that baseline from the snapshot's final active blocks and the net change
    # represented by trace events.
    final_active = sum(
        int(block.get("size", 0))
        for segment in snapshot.get("segments", [])
        for block in segment.get("blocks", [])
        if str(block.get("state", "")).startswith("active_")
    )
    trace_allocated = sum(
        int(event.get("size", 0))
        for event in traces[device]
        if event.get("action") == "alloc"
    )
    trace_freed = sum(
        int(event.get("size", 0))
        for event in traces[device]
        if event.get("action") == "free_completed"
    )
    baseline = max(final_active - trace_allocated + trace_freed, 0)

    active: dict[int, int] = {}
    total = baseline
    xs = [0]
    ys = [baseline]
    for index, event in enumerate(traces[device], start=1):
        action = event.get("action")
        address = event.get("addr")
        size = int(event.get("size", 0))
        if action == "alloc" and address is not None:
            address = int(address)
            active[address] = size
            total += size
        elif action in {"free_requested", "free_completed"} and address is not None:
            address = int(address)
            if action == "free_completed":
                # Allocations already live when recording began are part of
                # ``baseline`` rather than ``active``.  A free event still
                # carries its size, so subtract it even when no matching alloc
                # event appeared in this trace.
                total -= active.pop(address, size)
        xs.append(index)
        ys.append(max(total, 0))
    return xs, ys


def render(snapshot_path: Path, output: Path, *, device: int = 0, title: str | None = None) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as error:
        raise RuntimeError("matplotlib is required to render a timeline") from error

    with snapshot_path.open("rb") as handle:
        snapshot = pickle.load(handle)
    xs, ys = _active_series(snapshot, device=device)
    if not xs:
        raise ValueError("snapshot contains no device allocation events")

    output.parent.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(8, 4.4), dpi=140)
    axis.plot(xs, [value / 2**20 for value in ys], color="#2563eb", linewidth=1.8)
    axis.fill_between(xs, [value / 2**20 for value in ys], color="#93c5fd", alpha=0.35)
    axis.set_title(title or "Active Memory Timeline")
    axis.set_xlabel("allocator event index")
    axis.set_ylabel("active allocated memory (MiB)")
    axis.grid(True, alpha=0.25)
    peak = max(ys) / 2**20
    axis.text(
        0.99,
        0.96,
        f"peak active: {peak:.1f} MiB",
        transform=axis.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        bbox={"facecolor": "white", "alpha": 0.8, "edgecolor": "none"},
    )
    fig.tight_layout()
    fig.savefig(output, format="svg")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("snapshot", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--title")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    render(args.snapshot, args.output, device=args.device, title=args.title)
    print(f"wrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
