"""Render lightweight A2-K figures directly from measured CSV artifacts.

No result is embedded in this script: an image is created only when its source
CSV contains real ``success`` rows.  The two standard outputs are a checkpoint
time/memory trade-off and a FlashAttention latency/speedup comparison.
"""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

try:  # Support `python -m` and direct execution.
    from .common import default_output_dir, stderr
except ImportError:  # pragma: no cover - direct-script fallback.
    from common import default_output_dir, stderr  # type: ignore[no-redef]


@dataclass(frozen=True)
class CheckpointPoint:
    label: str
    context_length: int
    latency_ms: float
    peak_allocated_mib: float


@dataclass(frozen=True)
class FlashPoint:
    implementation: str
    sequence_length: int
    head_dim: int
    phase: str
    latency_ms: float
    speedup: float | None


def _float(row: dict[str, str], key: str) -> float | None:
    value = row.get(key, "").strip()
    try:
        return float(value) if value else None
    except ValueError:
        return None


def _integer(row: dict[str, str], key: str) -> int | None:
    value = _float(row, key)
    return int(value) if value is not None and value.is_integer() else None


def _success_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return [
            row
            for row in csv.DictReader(file)
            if row.get("status", "").strip() == "success" and row.get("formal", "").strip().lower() == "true"
        ]


def _checkpoint_points(path: Path) -> list[CheckpointPoint]:
    points: list[CheckpointPoint] = []
    for row in _success_rows(path):
        context_length = _integer(row, "context_length")
        latency = _float(row, "step_time_ms_p50")
        peak = _float(row, "peak_allocated_mib")
        block_size = row.get("checkpoint_block_size", "").strip()
        if context_length is None or latency is None or peak is None:
            continue
        label = "eager" if not block_size or block_size.lower() in {"none", "null"} else f"block={block_size}"
        points.append(CheckpointPoint(label=label, context_length=context_length, latency_ms=latency, peak_allocated_mib=peak))
    return points


def _flash_points(path: Path, *, phase: str) -> list[FlashPoint]:
    points: list[FlashPoint] = []
    for row in _success_rows(path):
        if row.get("phase", "").strip() != phase:
            continue
        sequence_length = _integer(row, "sequence_length")
        head_dim = _integer(row, "head_dim")
        latency = _float(row, "p50_ms")
        if sequence_length is None or head_dim is None or latency is None:
            continue
        points.append(
            FlashPoint(
                implementation=row.get("implementation", "unknown").strip() or "unknown",
                sequence_length=sequence_length,
                head_dim=head_dim,
                phase=phase,
                latency_ms=latency,
                speedup=_float(row, "speedup_vs_eager"),
            )
        )
    return points


def _save_checkpoint_plot(points: Sequence[CheckpointPoint], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    figure, axis = plt.subplots(figsize=(7.2, 4.8), constrained_layout=True)
    for point in points:
        marker = "o" if point.context_length == 1024 else "s"
        axis.scatter(point.latency_ms, point.peak_allocated_mib, marker=marker, s=70)
        axis.annotate(f"{point.label}, L={point.context_length}", (point.latency_ms, point.peak_allocated_mib), xytext=(5, 5), textcoords="offset points")
    axis.set_xlabel("Training-step p50 latency (ms)")
    axis.set_ylabel("Peak allocated memory (MiB)")
    axis.set_title("Activation checkpointing: measured time/memory trade-off")
    axis.grid(alpha=0.25)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def _group_flash(points: Iterable[FlashPoint]) -> dict[tuple[str, int], list[FlashPoint]]:
    grouped: dict[tuple[str, int], list[FlashPoint]] = defaultdict(list)
    for point in points:
        grouped[(point.implementation, point.head_dim)].append(point)
    for values in grouped.values():
        values.sort(key=lambda item: item.sequence_length)
    return dict(grouped)


def _save_flash_plot(points: Sequence[FlashPoint], output_path: Path, *, phase: str) -> None:
    import matplotlib.pyplot as plt

    figure, (latency_axis, speedup_axis) = plt.subplots(1, 2, figsize=(11.5, 4.6), constrained_layout=True)
    for (implementation, head_dim), values in _group_flash(points).items():
        label = f"{implementation}, D={head_dim}"
        sequences = [point.sequence_length for point in values]
        latency_axis.plot(sequences, [point.latency_ms for point in values], marker="o", label=label)
        speedup_values = [(point.sequence_length, point.speedup) for point in values if point.speedup is not None]
        if speedup_values:
            speedup_axis.plot(
                [sequence for sequence, _ in speedup_values],
                [speedup for _, speedup in speedup_values],
                marker="o",
                label=label,
            )
    for axis in (latency_axis, speedup_axis):
        axis.set_xscale("log", base=2)
        axis.set_xlabel("Sequence length")
        axis.grid(alpha=0.25)
        axis.legend(fontsize="small")
    latency_axis.set_ylabel("p50 latency (ms)")
    latency_axis.set_title(f"FlashAttention latency ({phase})")
    speedup_axis.axhline(1.0, color="black", linewidth=1, linestyle="--")
    speedup_axis.set_ylabel("Speedup vs equivalent eager baseline")
    speedup_axis.set_title(f"FlashAttention speedup ({phase})")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=160)
    plt.close(figure)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, default=default_output_dir(), help="Directory containing measured A2-K CSV files.")
    parser.add_argument("--output-dir", type=Path, default=None, help="Image directory (default: <input-dir>/assets).")
    parser.add_argument("--phase", choices=("forward", "backward", "forward_backward"), default="forward_backward")
    return parser


def run(*, input_dir: Path, output_dir: Path | None, phase: str) -> int:
    destination = output_dir if output_dir is not None else input_dir / "assets"
    created: list[Path] = []
    errors: list[str] = []

    checkpoint_csv = input_dir / "checkpointing.csv"
    if checkpoint_csv.exists():
        try:
            checkpoint_points = _checkpoint_points(checkpoint_csv)
            if checkpoint_points:
                output_path = destination / "checkpoint_tradeoff.png"
                _save_checkpoint_plot(checkpoint_points, output_path)
                created.append(output_path)
            else:
                errors.append("checkpointing.csv has no successful formal rows with p50 latency and peak allocated memory.")
        except (OSError, csv.Error, ValueError) as error:
            errors.append(f"checkpointing.csv could not be plotted: {type(error).__name__.lower()}")
    else:
        errors.append("checkpointing.csv is missing.")

    flash_csv = input_dir / "flash_benchmark.csv"
    if flash_csv.exists():
        try:
            flash_points = _flash_points(flash_csv, phase=phase)
            if flash_points:
                output_path = destination / "flash_latency_speedup.png"
                _save_flash_plot(flash_points, output_path, phase=phase)
                created.append(output_path)
            else:
                errors.append(f"flash_benchmark.csv has no successful formal {phase} rows.")
        except (OSError, csv.Error, ValueError) as error:
            errors.append(f"flash_benchmark.csv could not be plotted: {type(error).__name__.lower()}")
    else:
        errors.append("flash_benchmark.csv is missing.")

    if errors:
        stderr(" ".join(errors))
    if not created:
        stderr("No figure was created because no eligible measured rows were available.")
        return 2
    print("Created: " + ", ".join(str(path) for path in created))
    return 0 if not errors else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return run(input_dir=args.input_dir, output_dir=args.output_dir, phase=args.phase)


if __name__ == "__main__":
    raise SystemExit(main())
