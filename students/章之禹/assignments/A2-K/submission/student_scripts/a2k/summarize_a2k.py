"""Summarize isolated A2-K JSON results into public lightweight artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


CHECKPOINT_FIELDS = [
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
]
ATTENTION_FIELDS = [
    "implementation",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "causal",
    "phase",
    "samples",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "mean_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "speedup_vs_eager",
    "status",
    "error_type",
]
COMPILE_FIELDS = [
    "kind",
    "implementation",
    "model_size",
    "sequence_length",
    "head_dim",
    "num_layers",
    "batch_size",
    "dtype",
    "causal",
    "phase",
    "cold_start_ms",
    "samples",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "mean_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "speedup_vs_eager",
    "status",
    "error_type",
]
FLASH_FIELDS = [
    "implementation",
    "sequence_length",
    "head_dim",
    "batch_size",
    "dtype",
    "causal",
    "phase",
    "samples",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "mean_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "speedup_vs_eager",
    "q_tile_size",
    "k_tile_size",
    "num_warps",
    "num_stages",
    "status",
    "error_type",
]


def _read_json(path: Path) -> dict[str, Any] | None:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


def _json_value(value: Any) -> str:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False, separators=(",", ":"))
    if value is None:
        return ""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float) and not math.isfinite(value):
        return ""
    return str(value)


def _write_csv(path: Path, fields: list[str], rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: _json_value(row.get(field)) for field in fields})


def _status(payload: dict[str, Any]) -> str:
    value = payload.get("status")
    return str(value) if value else "failed"


def _checkpoint_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "checkpointing").glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        config = payload.get("config")
        config = config if isinstance(config, dict) else {}
        block = config.get("checkpoint_block_size")
        context = config.get("context_length")
        block_label = "none" if block in (None, 0, "0") else str(block)
        context_label = str(context or "unknown")
        rows.append(
            {
                "config_id": f"medium_ctx{context_label}_b1_block{block_label}",
                "model_size": config.get("model_size", "medium"),
                "num_layers": config.get("num_layers", 24),
                "context_length": context,
                "batch_size": config.get("batch_size", 1),
                "dtype": config.get("dtype", "bf16_autocast_fp32_params"),
                "checkpoint_block_size": block,
                "nested": config.get("nested", False),
                "warmup_steps": config.get("warmup_steps", 3),
                "measurement_steps": config.get("measurement_steps", 5),
                "step_time_ms_samples": payload.get("step_time_ms_samples", []),
                "step_time_ms_p50": payload.get("step_time_ms_p50"),
                "peak_allocated_mib": payload.get("peak_allocated_mib", 0.0),
                "peak_reserved_mib": payload.get("peak_reserved_mib", 0.0),
                "status": _status(payload),
            }
        )
    return rows


def _common_row(payload: dict[str, Any]) -> dict[str, Any]:
    experiment = payload.get("experiment")
    experiment = experiment if isinstance(experiment, dict) else {}
    row = dict(experiment)
    row.update(payload)
    row["status"] = _status(payload)
    return row


def _attention_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "attention_baseline").glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        row = _common_row(payload)
        row["implementation"] = "eager_explicit"
        rows.append(row)
    for row in rows:
        row["speedup_vs_eager"] = (
            1.0
            if row.get("status") == "complete" and row.get("p50_ms")
            else None
        )
    return rows


def _compile_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "compile").glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        row = _common_row(payload)
        rows.append(row)

    eager_p50: dict[tuple[str, str, int, int, str], float] = {}
    for row in rows:
        if (
            row.get("implementation") == "eager"
            and row.get("status") == "complete"
            and row.get("p50_ms") is not None
        ):
            key = (
                str(row.get("kind")),
                str(row.get("model_size") or ""),
                int(row.get("sequence_length", 0)),
                int(row.get("head_dim") or 0),
                str(row.get("phase")),
            )
            eager_p50[key] = float(row["p50_ms"])
    for row in rows:
        key = (
            str(row.get("kind")),
            str(row.get("model_size") or ""),
            int(row.get("sequence_length", 0)),
            int(row.get("head_dim") or 0),
            str(row.get("phase")),
        )
        baseline = eager_p50.get(key)
        row["speedup_vs_eager"] = (
            baseline / float(row["p50_ms"])
            if baseline and row.get("status") == "complete" and row.get("p50_ms")
            else None
        )
    return rows


def _flash_rows(root: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in sorted((root / "flash").glob("*.json")):
        payload = _read_json(path)
        if payload is None:
            continue
        row = _common_row(payload)
        rows.append(row)

    eager_p50: dict[tuple[int, int, str], float] = {}
    for row in rows:
        if (
            row.get("implementation") == "eager"
            and row.get("status") == "complete"
            and row.get("p50_ms") is not None
        ):
            key = (
                int(row.get("sequence_length", 0)),
                int(row.get("head_dim", 0)),
                str(row.get("phase")),
            )
            eager_p50[key] = float(row["p50_ms"])
    for row in rows:
        key = (
            int(row.get("sequence_length", 0)),
            int(row.get("head_dim", 0)),
            str(row.get("phase")),
        )
        baseline = eager_p50.get(key)
        row["speedup_vs_eager"] = (
            baseline / float(row["p50_ms"])
            if baseline and row.get("status") == "complete" and row.get("p50_ms")
            else None
        )
    return rows


def _first_payload(root: Path) -> dict[str, Any]:
    for path in sorted(root.rglob("*.json")):
        if path.name in {"memory_evidence.json", "run_metadata.json"}:
            continue
        payload = _read_json(path)
        if payload and payload.get("hardware"):
            return payload
    return {}


def _git_commit() -> str | None:
    try:
        return (
            subprocess.check_output(
                ["git", "rev-parse", "HEAD"],
                text=True,
                stderr=subprocess.DEVNULL,
            )
            .strip()
        )
    except (OSError, subprocess.SubprocessError):
        return None


def _write_metadata(root: Path, output: Path, rows: dict[str, list[dict[str, Any]]]) -> None:
    source = _first_payload(root)
    allocator = source.get("allocator", {})
    hardware = source.get("hardware", {})
    nvidia_smi = hardware.get("nvidia_smi", {}) if isinstance(hardware, dict) else {}
    counts = {
        name: {
            "rows": len(value),
            "complete": sum(row.get("status") == "complete" for row in value),
            "oom": sum(row.get("status") == "oom" for row in value),
            "failed": sum(row.get("status") == "failed" for row in value),
        }
        for name, value in rows.items()
    }
    metadata = {
        "assignment": "A2-K",
        "prompt_version": "26.1.4-k-rc.3",
        "starter_commit": "ca8bc81a59b70516f7ebb2da4808daade877c736",
        "implementation_commit_at_summary": _git_commit(),
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "seed": 20260722,
        "commands": [
            "python -m student_scripts.a2k.run_a2k --results local_results/formal "
            "--skip-checkpoint --skip-baseline --skip-compile --skip-correctness",
            "python -m student_scripts.a2k.correctness "
            "--output local_results/formal/correctness.json",
            "python -m pytest -q tests/test_attention.py -v",
            "python -m student_scripts.a2k.summarize_a2k "
            "--raw-results local_results/formal "
            "--output-dir public_results",
        ],
        "compile": {
            "backend": "inductor",
            "dynamic": False,
            "attention": "explicit PyTorch attention only",
            "small_model": "BasicsTransformerLM",
        },
        "hardware": {
            "gpu": hardware.get("gpu"),
            "observed_total_memory_mib": allocator.get("total_memory_mib"),
            "nvidia_smi": {
                "name": nvidia_smi.get("name"),
                "memory_total": nvidia_smi.get("memory_total"),
                "memory_free_at_start": nvidia_smi.get("memory_free_at_start"),
                "driver_version": nvidia_smi.get("driver_version"),
                "power_limit": nvidia_smi.get("power_limit"),
                "pstate": nvidia_smi.get("pstate"),
            },
            "torch": hardware.get("torch"),
            "cuda_runtime": hardware.get("cuda_runtime"),
            "python": hardware.get("python"),
            "triton": hardware.get("triton"),
        },
        "standard_target": {
            "gpu": "NVIDIA GeForce RTX 4090 24GB",
            "allocator_limit_mib": 23552,
            "hard_limit_mib": 24576,
            "observed_platform_note": (
                "The selected platform reports approximately 48 GiB on the 4090 "
                "notebook; every formal process still used the 23 GiB allocator guard."
            ),
        },
        "measurement": {
            "device": "cuda:0",
            "batch_size": 1,
            "dtype": "bf16",
            "causal": True,
            "attention_warmup_ms": 100,
            "attention_rep_ms": 300,
            "attention_quantiles": ["p20", "p50", "p80"],
            "cuda_synchronize": True,
            "single_process_serial": True,
            "tf32_matmul": False,
            "tf32_cudnn": False,
        },
        "allocator": allocator,
        "result_counts": counts,
    }
    output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_memory_evidence(root: Path, output: Path) -> None:
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob("*.json")):
        if path.name in {"memory_evidence.json", "run_metadata.json"}:
            continue
        payload = _read_json(path)
        if payload is None:
            continue
        if "peak_allocated_mib" in payload or "peak_reserved_mib" in payload:
            records.append(
                {
                    "source": path.name,
                    "peak_allocated_mib": float(payload.get("peak_allocated_mib", 0.0) or 0.0),
                    "peak_reserved_mib": float(payload.get("peak_reserved_mib", 0.0) or 0.0),
                    "status": _status(payload),
                }
            )
    source = _first_payload(root)
    allocator = source.get("allocator", {})
    max_allocated = max((item["peak_allocated_mib"] for item in records), default=0.0)
    max_reserved = max((item["peak_reserved_mib"] for item in records), default=0.0)
    payload = {
        "allocator": {
            "allocator_fraction": float(allocator.get("allocator_fraction", 1.0)),
            "allocator_limit_mib": 23552,
        },
        "hard_limit_mib": 24576,
        "pytorch_peak_allocated_mib": max_allocated,
        "pytorch_peak_reserved_mib": max_reserved,
        "within_24gib": bool(max_reserved <= 23552 and max_allocated <= 24576),
        "records": records,
    }
    output.write_text(
        json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _svg_escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _write_simple_svg(path: Path, title: str, labels: list[str], values: list[float], ylabel: str) -> None:
    width = max(960, 95 * len(values) + 110)
    height = 560
    left, right, top, bottom = 80, 30, 70, 100
    plot_w, plot_h = width - left - right, height - top - bottom
    finite = [value for value in values if math.isfinite(value) and value >= 0]
    ymax = max(finite, default=1.0) * 1.15 or 1.0
    bar_gap = plot_w / max(len(values), 1)
    bar_w = max(4.0, bar_gap * 0.68)
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text x="{left}" y="34" font-family="sans-serif" font-size="22" font-weight="600">{_svg_escape(title)}</text>',
        f'<text x="18" y="{top + plot_h / 2}" transform="rotate(-90 18 {top + plot_h / 2})" font-family="sans-serif" font-size="14">{_svg_escape(ylabel)}</text>',
        f'<line x1="{left}" y1="{top + plot_h}" x2="{width-right}" y2="{top + plot_h}" stroke="#333"/>',
        f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top + plot_h}" stroke="#333"/>',
    ]
    for index, (label, value) in enumerate(zip(labels, values, strict=False)):
        x = left + index * bar_gap + (bar_gap - bar_w) / 2
        if math.isfinite(value) and value >= 0:
            bar_h = plot_h * value / ymax
            y = top + plot_h - bar_h
            parts.append(
                f'<rect x="{x:.2f}" y="{y:.2f}" width="{bar_w:.2f}" height="{bar_h:.2f}" fill="#3973ac"/>'
            )
            parts.append(
                f'<text x="{x + bar_w / 2:.2f}" y="{max(top + 14, y - 5):.2f}" text-anchor="middle" font-family="sans-serif" font-size="11">{value:.1f}</text>'
            )
        parts.append(
            f'<text x="{x + bar_w / 2:.2f}" y="{top + plot_h + 24}" text-anchor="end" transform="rotate(-35 {x + bar_w / 2:.2f} {top + plot_h + 24})" font-family="sans-serif" font-size="11">{_svg_escape(label)}</text>'
        )
    parts.append(
        f'<text x="{left}" y="{height - 18}" font-family="sans-serif" font-size="12">OOM/failed rows are omitted from bars and retained in CSV.</text>'
    )
    parts.append("</svg>\n")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts), encoding="utf-8")


def _write_charts(output_dir: Path, checkpoint_rows: list[dict[str, Any]], flash_rows: list[dict[str, Any]]) -> None:
    successful_checkpoints = [
        row for row in checkpoint_rows if row.get("status") == "complete" and row.get("context_length") == 1024
    ]
    successful_checkpoints.sort(key=lambda row: (int(row.get("checkpoint_block_size") or 0),))
    _write_simple_svg(
        output_dir / "checkpoint_memory_latency.svg",
        "Checkpointing: context 1024 peak allocated memory (MiB)",
        [
            f"block {row.get('checkpoint_block_size') or 'none'}"
            for row in successful_checkpoints
        ],
        [float(row.get("peak_allocated_mib") or 0.0) for row in successful_checkpoints],
        "peak allocated MiB",
    )

    grouped: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    for row in flash_rows:
        if row.get("status") == "complete" and row.get("phase") == "forward":
            grouped[(int(row.get("sequence_length", 0)), int(row.get("head_dim", 0)))].append(row)
    selected: list[dict[str, Any]] = []
    for key in ((512, 64), (2048, 128), (8192, 128), (16384, 64), (16384, 128)):
        selected.extend(grouped.get(key, []))
    implementation_order = {"eager": 0, "compiled": 1, "triton": 2}
    selected.sort(
        key=lambda row: (
            int(row.get("sequence_length", 0)),
            int(row.get("head_dim", 0)),
            implementation_order.get(str(row.get("implementation")), 99),
        )
    )
    _write_simple_svg(
        output_dir / "flash_forward_latency.svg",
        "Flash forward p50 latency across core and 16384 boundary shapes",
        [
            f"{row.get('sequence_length')}x{row.get('head_dim')} {row.get('implementation')}"
            for row in selected
        ],
        [float(row.get("p50_ms") or 0.0) for row in selected],
        "p50 latency (ms)",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-results", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_rows = _checkpoint_rows(args.raw_results)
    attention_rows = _attention_rows(args.raw_results)
    compile_rows = _compile_rows(args.raw_results)
    flash_rows = _flash_rows(args.raw_results)
    rows = {
        "checkpointing": checkpoint_rows,
        "attention_baseline": attention_rows,
        "compile_comparison": compile_rows,
        "flash_benchmark": flash_rows,
    }
    _write_csv(output_dir / "checkpointing.csv", CHECKPOINT_FIELDS, checkpoint_rows)
    _write_csv(output_dir / "attention_baseline.csv", ATTENTION_FIELDS, attention_rows)
    _write_csv(output_dir / "compile_comparison.csv", COMPILE_FIELDS, compile_rows)
    _write_csv(output_dir / "flash_benchmark.csv", FLASH_FIELDS, flash_rows)
    _write_memory_evidence(args.raw_results, output_dir / "memory_evidence.json")
    _write_metadata(args.raw_results, output_dir / "run_metadata.json", rows)
    _write_charts(output_dir.parent / "assets", checkpoint_rows, flash_rows)
    print(
        json.dumps(
            {name: len(value) for name, value in rows.items()},
            indent=2,
            ensure_ascii=False,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
