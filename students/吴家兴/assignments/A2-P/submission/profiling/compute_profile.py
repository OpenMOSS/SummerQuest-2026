#!/usr/bin/env python3
"""Capture six stable train-step traces with torch.profiler."""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

import torch
from einops import einsum
from torch.autograd import DeviceType
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
)

try:
    from .common import (
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        release_cuda,
        run_step,
        stage_elapsed_ms,
        write_json,
    )
    from .config import MODEL_CONFIGS, RunConfig
except ImportError:
    from common import (
        build_batch,
        build_model,
        build_optimizer,
        memory_stats_mib,
        public_environment,
        release_cuda,
        run_step,
        stage_elapsed_ms,
        write_json,
    )
    from config import MODEL_CONFIGS, RunConfig


STAGE_NAMES = (
    "profile/warmup",
    "profile/measure",
    "zero_grad",
    "forward",
    "backward",
    "optimizer",
    "attention/scores",
    "attention/softmax",
    "attention/value",
)

SUMMARY_FIELDS = (
    "run_id",
    "model_size",
    "batch_size",
    "context_length",
    "mode",
    "dtype",
    "row_type",
    "name",
    "calls",
    "cpu_total_us",
    "cuda_total_us",
    "cuda_self_us",
    "rank",
)


def annotated_scaled_dot_product_attention(
    Q: torch.Tensor,
    K: torch.Tensor,
    V: torch.Tensor,
    mask: torch.Tensor | None = None,
) -> torch.Tensor:
    """Starter-equivalent attention with only profiler annotations added."""

    from cs336_basics.nn_utils import softmax

    with record_function("attention/scores"):
        scores = einsum(
            Q,
            K,
            "... queries d_k, ... keys d_k -> ... queries keys",
        )
        scores = scores / math.sqrt(Q.shape[-1])
        if mask is not None:
            scores = torch.where(mask, scores, -torch.inf)
    with record_function("attention/softmax"):
        attention = softmax(scores, dim=-1)
    with record_function("attention/value"):
        output = einsum(
            attention,
            V,
            "... queries keys, ... keys d_v -> ... queries d_v",
        )
    return output


@contextmanager
def annotated_attention() -> Iterator[None]:
    """Temporarily install annotations without changing the starter on disk."""

    import cs336_basics.model as model_module

    original = model_module.scaled_dot_product_attention
    model_module.scaled_dot_product_attention = (
        annotated_scaled_dot_product_attention
    )
    try:
        yield
    finally:
        model_module.scaled_dot_product_attention = original


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _device_time(event: Any) -> float:
    return float(getattr(event, "device_time_total", 0.0) or 0.0)


def _self_device_time(event: Any) -> float:
    return float(getattr(event, "self_device_time_total", 0.0) or 0.0)


def _summary_rows(
    profiler: Any,
    config: RunConfig,
    measured_stage_ms: dict[str, float],
    top_ops: int,
    top_kernels: int,
) -> list[dict[str, Any]]:
    base = {
        "run_id": config.run_id,
        "model_size": config.model_size,
        "batch_size": config.batch_size,
        "context_length": config.context_length,
        "mode": config.mode,
        "dtype": config.dtype,
    }
    averages = list(profiler.key_averages())
    by_name: dict[str, list[Any]] = defaultdict(list)
    for event in averages:
        by_name[event.key].append(event)
    rows: list[dict[str, Any]] = []

    for rank, name in enumerate(STAGE_NAMES, start=1):
        events = by_name.get(name)
        if not events:
            continue
        calls = max(int(event.count) for event in events)
        cpu_total = max(float(event.cpu_time_total) for event in events)
        cuda_total = max(_device_time(event) for event in events)
        cuda_self = max(_self_device_time(event) for event in events)
        if name in measured_stage_ms:
            cuda_total = measured_stage_ms[name] * 1000
        rows.append(
            {
                **base,
                "row_type": "stage_range",
                "name": name,
                "calls": calls,
                "cpu_total_us": round(cpu_total, 3),
                "cuda_total_us": round(cuda_total, 3),
                "cuda_self_us": round(cuda_self, 3),
                "rank": rank,
            }
        )

    operator_events = [
        event
        for event in averages
        if event.key not in STAGE_NAMES
        and (
            event.key.startswith("aten::")
            or event.key.startswith("autograd::")
        )
        and float(event.cpu_time_total) > 0
    ]
    operator_events.sort(
        key=lambda event: (_device_time(event), event.cpu_time_total),
        reverse=True,
    )
    for rank, event in enumerate(operator_events[:top_ops], start=1):
        rows.append(
            {
                **base,
                "row_type": "operator",
                "name": event.key,
                "calls": int(event.count),
                "cpu_total_us": round(float(event.cpu_time_total), 3),
                "cuda_total_us": round(_device_time(event), 3),
                "cuda_self_us": round(_self_device_time(event), 3),
                "rank": rank,
            }
        )

    kernel_totals: dict[str, dict[str, float]] = defaultdict(
        lambda: {"calls": 0, "cuda_total_us": 0.0}
    )
    for event in profiler.events():
        if (
            event.device_type != DeviceType.CUDA
            or event.name in STAGE_NAMES
            or bool(getattr(event, "is_user_annotation", False))
        ):
            continue
        item = kernel_totals[event.name]
        item["calls"] += 1
        item["cuda_total_us"] += _device_time(event)
    ordered_kernels = sorted(
        kernel_totals.items(),
        key=lambda item: item[1]["cuda_total_us"],
        reverse=True,
    )
    for rank, (name, totals) in enumerate(
        ordered_kernels[:top_kernels],
        start=1,
    ):
        rows.append(
            {
                **base,
                "row_type": "cuda_kernel_or_activity",
                "name": name,
                "calls": int(totals["calls"]),
                "cpu_total_us": 0.0,
                "cuda_total_us": round(totals["cuda_total_us"], 3),
                "cuda_self_us": round(totals["cuda_total_us"], 3),
                "rank": rank,
            }
        )
    return rows


def profile_once(
    config: RunConfig,
    trace_dir: Path,
    top_ops: int,
    top_kernels: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    config.validate()
    model = build_model(config)
    inputs, targets = build_batch(config)
    optimizer = build_optimizer(model)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    with annotated_attention():
        for _ in range(config.warmup_steps):
            run_step(
                model,
                inputs,
                targets,
                "train_step",
                config.dtype,
                optimizer,
            )
            torch.cuda.synchronize()

        torch.cuda.reset_peak_memory_stats()
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=0, warmup=1, active=1, repeat=1),
            record_shapes=False,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            # The first profiler-only warm-up step contains no model work.
            # Workload warm-up has already run outside capture.
            with record_function("profiler/internal_warmup"):
                pass
            profiler.step()
            # This zero-work marker exposes the external warm-up/capture
            # boundary in the one active trace step.
            with record_function("profile/warmup"):
                pass
            with record_function("profile/measure"):
                result = run_step(
                    model,
                    inputs,
                    targets,
                    "train_step",
                    config.dtype,
                    optimizer,
                    capture_stage_events=True,
                )
            profiler.step()
        torch.cuda.synchronize()

    measured_stage_ms = stage_elapsed_ms(result.stage_events)
    measured_stage_ms["profile/measure"] = sum(measured_stage_ms.values())
    trace_dir.mkdir(parents=True, exist_ok=True)
    trace_path = trace_dir / f"{config.run_id}.trace.json"
    profiler.export_chrome_trace(str(trace_path))
    rows = _summary_rows(
        profiler,
        config,
        measured_stage_ms,
        top_ops,
        top_kernels,
    )
    metadata = {
        "run_id": config.run_id,
        "config": config.as_dict(),
        "parameter_count": parameter_count,
        "profiler_schedule": {
            "external_warmup_steps": config.warmup_steps,
            "wait": 0,
            "warmup": 1,
            "active": 1,
            "repeat": 1,
        },
        "profile_warmup_range": "zero-work boundary marker; warm-up ran before capture",
        "captured_measurement_steps": 1,
        "stage_cuda_ms": measured_stage_ms,
        "loss": result.loss,
        "memory": {
            key: round(value, 3) for key, value in memory_stats_mib().items()
        },
        "local_trace": {
            "file": trace_path.name,
            "bytes": trace_path.stat().st_size,
            "sha256": _sha256(trace_path),
            "committed": False,
        },
    }
    del model, inputs, targets, optimizer, profiler
    release_cuda()
    return rows, metadata


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", default=["small", "medium"])
    parser.add_argument(
        "--contexts",
        nargs="+",
        type=int,
        default=[256, 512, 1024],
    )
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--top-ops", type=int, default=15)
    parser.add_argument("--top-kernels", type=int, default=15)
    parser.add_argument("--trace-dir", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    invalid = [name for name in args.models if name not in MODEL_CONFIGS]
    if invalid:
        raise ValueError(f"unknown model sizes: {invalid}")

    all_rows: list[dict[str, Any]] = []
    run_metadata: list[dict[str, Any]] = []
    for model_size in args.models:
        for context_length in args.contexts:
            config = RunConfig(
                model_size=model_size,
                batch_size=args.batch_size,
                context_length=context_length,
                mode="train_step",
                dtype=args.dtype,
                warmup_steps=args.warmup,
                measurement_steps=1,
                seed=args.seed,
            )
            print(f"profiling {config.run_id}", flush=True)
            rows, metadata = profile_once(
                config,
                args.trace_dir,
                args.top_ops,
                args.top_kernels,
            )
            all_rows.extend(rows)
            run_metadata.append(metadata)

    args.summary.parent.mkdir(parents=True, exist_ok=True)
    with args.summary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=SUMMARY_FIELDS, lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(all_rows)
    write_json(
        args.metadata,
        {
            "schema_version": 1,
            "experiment": "compute_profiling",
            "tool": "torch.profiler",
            "activities": ["CPU", "CUDA"],
            "annotation_ranges": list(STAGE_NAMES),
            "trace_policy": "raw Chrome traces retained locally and not committed",
            "environment": public_environment(),
            "runs": run_metadata,
        },
    )
    print(f"wrote {len(all_rows)} summary rows for {len(run_metadata)} traces")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
