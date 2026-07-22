#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
import time
from contextlib import nullcontext
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
    sample_statistics,
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
    write_json,
)
from profiling.nvtx_ranges import instrument_attention, stage_range


MODES = ("forward", "forward_backward", "train_step")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P end-to-end benchmark and profiler")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=MODES, default="train_step")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument("--run-name")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--profiler", choices=("none", "torch"), default="none")
    parser.add_argument("--trace-output", type=Path)
    parser.add_argument("--operator-summary-output", type=Path)
    parser.add_argument("--nvtx", action="store_true")

    # These overrides make CPU smoke tests inexpensive without changing formal model sizes.
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--d-model", type=int)
    parser.add_argument("--d-ff", type=int)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--num-heads", type=int)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.batch_size < 1 or args.context_length < 1:
        raise SystemExit("batch size and context length must be positive")
    if args.warmup < 0 or args.steps < 1:
        raise SystemExit("warmup must be non-negative and steps must be positive")
    if args.profiler == "torch" and args.steps != 1:
        raise SystemExit("torch profiler runs must use --steps 1 to capture one stable step")
    if args.profiler == "torch" and args.trace_output is None:
        raise SystemExit("torch profiler runs require --trace-output")
    if args.trace_output is not None and args.profiler != "torch":
        raise SystemExit("--trace-output requires --profiler torch")
    if args.operator_summary_output is not None and args.profiler != "torch":
        raise SystemExit("--operator-summary-output requires --profiler torch")


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
    return slugify(f"{args.model_size}-bs{args.batch_size}-ctx{args.context_length}-{args.mode}-{args.dtype}-w{args.warmup}-n{args.steps}-seed{args.seed}")


def build_model_and_data(
    args: argparse.Namespace, device: torch.device, config: dict[str, int]
) -> tuple[BasicsTransformerLM, torch.optim.Optimizer | None, torch.Tensor, torch.Tensor]:
    set_seed(args.seed)
    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        **config,
    ).to(device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters()) if args.mode == "train_step" else None
    tokens = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    labels = torch.randint(
        args.vocab_size,
        (args.batch_size, args.context_length),
        device=device,
    )
    return model, optimizer, tokens, labels


def run_step(
    *,
    args: argparse.Namespace,
    device: torch.device,
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer | None,
    tokens: torch.Tensor,
    labels: torch.Tensor,
    instrument: bool,
) -> torch.Tensor:
    range_kwargs = {"device": device, "enabled": instrument, "use_nvtx": args.nvtx}
    if args.mode == "forward":
        with torch.no_grad(), stage_range("forward", **range_kwargs), autocast_context(device, args.dtype):
            return model(tokens)

    if args.mode == "train_step":
        assert optimizer is not None
        with stage_range("zero_grad", **range_kwargs):
            optimizer.zero_grad(set_to_none=True)
    else:
        with stage_range("zero_grad", **range_kwargs):
            model.zero_grad(set_to_none=True)

    with stage_range("forward", **range_kwargs), autocast_context(device, args.dtype):
        logits = model(tokens)
    with stage_range("loss", **range_kwargs):
        loss = F.cross_entropy(logits.reshape(-1, args.vocab_size).float(), labels.reshape(-1))
    with stage_range("backward", **range_kwargs):
        loss.backward()
    if args.mode == "train_step":
        assert optimizer is not None
        with stage_range("optimizer", **range_kwargs):
            optimizer.step()
    return loss.detach()


def timed_step(**kwargs: Any) -> float:
    device: torch.device = kwargs["device"]
    synchronize(device)
    start = time.perf_counter()
    run_step(**kwargs)
    synchronize(device)
    return (time.perf_counter() - start) * 1000


def profiler_context(args: argparse.Namespace, device: torch.device):
    if args.profiler == "none":
        return nullcontext(None)
    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)
    return torch.profiler.profile(
        activities=activities,
        record_shapes=True,
        profile_memory=False,
        with_stack=False,
    )


def profiler_metric(event: Any, *names: str) -> float:
    for name in names:
        value = getattr(event, name, None)
        if value is not None:
            return float(value)
    return 0.0


def operator_summary(profiler: Any) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for event in profiler.key_averages():
        rows.append(
            {
                "name": str(event.key),
                "calls": int(event.count),
                "cpu_total_us": profiler_metric(event, "cpu_time_total"),
                "cpu_self_us": profiler_metric(event, "self_cpu_time_total"),
                "device_total_us": profiler_metric(event, "device_time_total", "cuda_time_total"),
                "device_self_us": profiler_metric(event, "self_device_time_total", "self_cuda_time_total"),
            }
        )
    return rows


def main() -> int:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA is unavailable; use --device cpu only for smoke tests")
    if args.dtype == "bf16" and device.type == "cuda" and not torch.cuda.is_bf16_supported():
        raise SystemExit("the selected CUDA device does not support BF16")

    config = resolved_model_config(args)
    run_name = slugify(args.run_name) if args.run_name else default_run_name(args)
    metadata_output = args.metadata_output or args.output.with_name("benchmark_metadata.json")
    model, optimizer, tokens, labels = build_model_and_data(args, device, config)
    instrument = args.profiler != "none" or args.nvtx
    range_kwargs = {"device": device, "enabled": instrument, "use_nvtx": args.nvtx}

    timings: list[float] = []
    with instrument_attention(device=device, enabled=instrument, use_nvtx=args.nvtx):
        with stage_range("profile/warmup", **range_kwargs):
            for _ in range(args.warmup):
                run_step(
                    args=args,
                    device=device,
                    model=model,
                    optimizer=optimizer,
                    tokens=tokens,
                    labels=labels,
                    instrument=instrument,
                )
                synchronize(device)

        with profiler_context(args, device) as profiler:
            with stage_range("profile/measure", **range_kwargs):
                for _ in range(args.steps):
                    timings.append(
                        timed_step(
                            args=args,
                            device=device,
                            model=model,
                            optimizer=optimizer,
                            tokens=tokens,
                            labels=labels,
                            instrument=instrument,
                        )
                    )
            if profiler is not None:
                profiler.step()

    if args.profiler == "torch":
        assert args.trace_output is not None
        operator_output = args.operator_summary_output or args.trace_output.with_suffix(".ops.json")
        args.trace_output.parent.mkdir(parents=True, exist_ok=True)
        profiler.export_chrome_trace(str(args.trace_output))
        write_json(operator_output, operator_summary(profiler))
    else:
        operator_output = None

    stats = sample_statistics(timings)
    environment = environment_metadata(device)
    timestamp = utc_timestamp()
    common_row = {
        "run_id": run_name,
        "timestamp_utc": timestamp,
        "model_size": args.model_size,
        **config,
        "vocab_size": args.vocab_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": args.dtype,
        "seed": args.seed,
        "warmup": args.warmup,
        "steps": args.steps,
        "device": str(device),
        "gpu_name": environment["gpu_name"],
        "torch_version": environment["torch_version"],
        "compiled_cuda": environment["compiled_cuda"],
        "profiler": args.profiler,
        **stats,
    }
    rows = [{**common_row, "measurement_step": step, "time_ms": timing} for step, timing in enumerate(timings)]
    upsert_csv_rows(args.output, rows)

    metadata = {
        "run_id": run_name,
        "timestamp_utc": timestamp,
        "status": "success",
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
            "steps": args.steps,
        },
        "environment": environment,
        "profiler": args.profiler,
        "artifacts": {
            "timings_csv": artifact_name(args.output),
            "trace": artifact_name(args.trace_output),
            "operator_summary": artifact_name(operator_output),
        },
        "statistics": stats,
    }
    upsert_json_record(metadata_output, metadata)

    print(f"{run_name}: mean={stats['mean_ms']:.3f} ms sample_std={stats['sample_std_ms']:.3f} ms cv={stats['cv']:.6f}")
    print(f"timings={args.output} metadata={metadata_output}")
    if args.trace_output is not None:
        print(f"trace={args.trace_output}")
        print(f"operator_summary={operator_output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
