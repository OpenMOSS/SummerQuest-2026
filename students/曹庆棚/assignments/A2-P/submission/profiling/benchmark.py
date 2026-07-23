from __future__ import annotations

import argparse
import hashlib
import math
import random
import statistics
import sys
import timeit
from contextlib import AbstractContextManager, nullcontext
from contextlib import contextmanager
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from profiling.config import (
    DEFAULT_LR,
    DEFAULT_SEED,
    DEFAULT_WEIGHT_DECAY,
    DTYPES,
    MODEL_CONFIGS,
    MODES,
    VOCAB_SIZE,
    base_metadata,
    classify_error,
    environment_metadata,
    make_run_name,
    model_config_dict,
    public_relative_path,
    safe_error_summary,
    utc_now,
    write_json,
)
from profiling.nvtx_ranges import patched_attention_ranges, stage_range


@dataclass
class ExperimentState:
    model: BasicsTransformerLM
    optimizer: AdamW | None
    input_ids: torch.Tensor
    targets: torch.Tensor
    device: torch.device
    mode: str
    dtype: str
    model_size: str
    context_length: int
    batch_size: int
    seed: int
    parameter_count: int
    model_fingerprint: str


@dataclass
class StepResult:
    loss: torch.Tensor | None = None
    output_probe: torch.Tensor | None = None


def configure_runtime(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = False


def autocast_context(dtype: str, device: torch.device) -> AbstractContextManager[Any]:
    if dtype == "fp32":
        return nullcontext()
    if dtype != "bf16":
        raise ValueError(f"unsupported dtype: {dtype}")
    if device.type != "cuda":
        raise ValueError("BF16 autocast experiments require a CUDA device")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("the selected CUDA device does not report BF16 support")
    return torch.autocast(device_type="cuda", dtype=torch.bfloat16)


def _fingerprint_model(model: torch.nn.Module, sample_values: int = 256) -> str:
    digest = hashlib.sha256()
    remaining = sample_values
    with torch.no_grad():
        for parameter in model.parameters():
            if remaining <= 0:
                break
            flat = parameter.detach().reshape(-1)
            count = min(remaining, flat.numel())
            values = flat[:count].to(device="cpu", dtype=torch.float32).numpy()
            digest.update(values.tobytes())
            remaining -= count
    return digest.hexdigest()


def build_experiment(
    *,
    model_size: str,
    batch_size: int,
    context_length: int,
    mode: str,
    dtype: str,
    seed: int,
    device_name: str,
    learning_rate: float,
    weight_decay: float,
) -> ExperimentState:
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"unknown model size: {model_size}")
    if mode not in MODES:
        raise ValueError(f"unknown mode: {mode}")
    if dtype not in DTYPES:
        raise ValueError(f"unknown dtype: {dtype}")
    if batch_size <= 0 or context_length <= 0:
        raise ValueError("batch size and context length must be positive")

    device = torch.device(device_name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
    if device.type == "cuda":
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)

    configure_runtime(seed)
    model_config = MODEL_CONFIGS[model_size]
    model_kwargs = {
        "vocab_size": VOCAB_SIZE,
        "context_length": context_length,
        "d_model": model_config.d_model,
        "num_layers": model_config.num_layers,
        "num_heads": model_config.num_heads,
        "d_ff": model_config.d_ff,
    }

    # Construct directly on CUDA to avoid a second full XL-sized CPU copy.
    if device.type == "cuda":
        with torch.device(device):
            model = BasicsTransformerLM(**model_kwargs)
    else:
        model = BasicsTransformerLM(**model_kwargs).to(device)
    model.train(mode != "forward")

    input_ids = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        low=0,
        high=VOCAB_SIZE,
        size=(batch_size, context_length),
        device=device,
        dtype=torch.long,
    )

    optimizer = None
    if mode == "train_step":
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    return ExperimentState(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        device=device,
        mode=mode,
        dtype=dtype,
        model_size=model_size,
        context_length=context_length,
        batch_size=batch_size,
        seed=seed,
        parameter_count=parameter_count,
        model_fingerprint=_fingerprint_model(model),
    )


def run_step(
    state: ExperimentState,
    *,
    outer_label: str | None = None,
    capture_probe: bool = False,
    synchronize_at_end: bool = False,
    stage_callback: Callable[[str], None] | None = None,
    cuda_stage_events: dict[str, list[tuple[torch.cuda.Event, torch.cuda.Event]]] | None = None,
) -> StepResult:
    def enter_stage(name: str) -> None:
        if stage_callback is not None:
            stage_callback(name)

    @contextmanager
    def measured_stage(name: str):
        enter_stage(name)
        event_pair = None
        if cuda_stage_events is not None and state.device.type == "cuda":
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            event_pair = (start_event, end_event)
        with stage_range(name):
            yield
        if event_pair is not None:
            event_pair[1].record()
            assert cuda_stage_events is not None
            cuda_stage_events.setdefault(name, []).append(event_pair)

    outer = stage_range(outer_label) if outer_label else nullcontext()
    with outer:
        if state.mode != "forward":
            with measured_stage("zero_grad"):
                if state.optimizer is not None:
                    state.optimizer.zero_grad(set_to_none=True)
                else:
                    state.model.zero_grad(set_to_none=True)

        if state.mode == "forward":
            with torch.no_grad(), measured_stage("forward"), autocast_context(state.dtype, state.device):
                logits = state.model(state.input_ids)
            loss = None
        else:
            with measured_stage("forward"), autocast_context(state.dtype, state.device):
                logits = state.model(state.input_ids)
            with measured_stage("loss"), autocast_context(state.dtype, state.device):
                loss = cross_entropy(
                    logits.reshape(-1, VOCAB_SIZE),
                    state.targets.reshape(-1),
                )
            with measured_stage("backward"):
                loss.backward()
            if state.mode == "train_step":
                assert state.optimizer is not None
                with measured_stage("optimizer"):
                    state.optimizer.step()

        if synchronize_at_end and state.device.type == "cuda":
            torch.cuda.synchronize(state.device)

    return StepResult(
        loss=loss.detach() if loss is not None else None,
        output_probe=logits.detach() if capture_probe else None,
    )


def perform_warmup(
    state: ExperimentState,
    steps: int,
    *,
    label: bool = True,
    stage_callback: Callable[[str], None] | None = None,
) -> None:
    if steps < 0:
        raise ValueError("warm-up steps cannot be negative")
    context = stage_range("profile/warmup") if label and steps else nullcontext()
    with context:
        for _ in range(steps):
            run_step(state, stage_callback=stage_callback)
            if state.device.type == "cuda":
                torch.cuda.synchronize(state.device)


def benchmark_state(
    state: ExperimentState,
    *,
    warmup: int,
    steps: int,
    stage_callback: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    if steps <= 0:
        raise ValueError("measurement steps must be positive")
    perform_warmup(state, warmup, stage_callback=stage_callback)
    if state.device.type == "cuda":
        torch.cuda.synchronize(state.device)
        torch.cuda.reset_peak_memory_stats(state.device)

    timings_ms: list[float] = []
    measurement_losses: list[float] = []
    last_result: StepResult | None = None
    for step_index in range(steps):
        if state.device.type == "cuda":
            torch.cuda.synchronize(state.device)
        start = timeit.default_timer()
        last_result = run_step(
            state,
            capture_probe=step_index == steps - 1,
            stage_callback=stage_callback,
        )
        if state.device.type == "cuda":
            torch.cuda.synchronize(state.device)
        elapsed = timeit.default_timer() - start
        timings_ms.append(elapsed * 1000.0)
        if last_result.loss is not None:
            measurement_losses.append(float(last_result.loss.float().item()))

    assert last_result is not None
    last_loss = float(last_result.loss.float().item()) if last_result.loss is not None else None
    logits_finite = None
    if last_result.output_probe is not None:
        sample = last_result.output_probe.reshape(-1)[:4096]
        logits_finite = bool(torch.isfinite(sample).all().item())

    mean_ms = statistics.fmean(timings_ms)
    sample_std_ms = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    peak_memory = {
        "peak_allocated_bytes": None,
        "peak_reserved_bytes": None,
        "peak_allocated_mib": None,
        "peak_reserved_mib": None,
    }
    if state.device.type == "cuda":
        peak_allocated = torch.cuda.max_memory_allocated(state.device)
        peak_reserved = torch.cuda.max_memory_reserved(state.device)
        peak_memory = {
            "peak_allocated_bytes": peak_allocated,
            "peak_reserved_bytes": peak_reserved,
            "peak_allocated_mib": round(peak_allocated / (1024**2), 4),
            "peak_reserved_mib": round(peak_reserved / (1024**2), 4),
        }
    return {
        "timings_ms": timings_ms,
        "measurement_losses": measurement_losses,
        "losses_finite": all(math.isfinite(value) for value in measurement_losses),
        "summary": {
            "mean_ms": mean_ms,
            "sample_std_ms": sample_std_ms,
            "cv": sample_std_ms / mean_ms if mean_ms else math.nan,
            "min_ms": min(timings_ms),
            "median_ms": statistics.median(timings_ms),
            "max_ms": max(timings_ms),
        },
        "last_loss": last_loss,
        "logits_finite": logits_finite,
        **peak_memory,
    }


def add_common_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--run-id", default="manual")
    parser.add_argument("--model-size", choices=tuple(MODEL_CONFIGS), required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=MODES, required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=DTYPES, default="fp32")
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--learning-rate", type=float, default=DEFAULT_LR)
    parser.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY)
    parser.add_argument(
        "--annotate-attention",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="enable attention record_function ranges; keep disabled for clean wall-clock benchmarks",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P end-to-end benchmark")
    add_common_arguments(parser)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def run_benchmark(args: argparse.Namespace) -> dict[str, Any]:
    run_name = make_run_name(
        "benchmark",
        args.model_size,
        args.batch_size,
        args.context_length,
        args.mode,
        args.dtype,
        "timer",
    )
    payload = base_metadata(args.run_id, run_name, "python_timer")
    payload.update(
        {
            "kind": "benchmark_result",
            "model_size": args.model_size,
            "model_config": model_config_dict(args.model_size),
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "mode": args.mode,
            "dtype": args.dtype,
            "autocast_dtype": "bfloat16" if args.dtype == "bf16" else None,
            "warmup_steps": args.warmup,
            "measurement_steps": args.steps,
            "seed": args.seed,
            "optimizer": "cs336_basics.optimizer.AdamW" if args.mode == "train_step" else None,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "output_file": public_relative_path(args.output),
            "timer": "timeit.default_timer",
            "synchronize_before_and_after_step": args.device.startswith("cuda"),
            "attention_instrumentation": args.annotate_attention,
        }
    )

    failure_stage = "initialization"
    try:
        state = build_experiment(
            model_size=args.model_size,
            batch_size=args.batch_size,
            context_length=args.context_length,
            mode=args.mode,
            dtype=args.dtype,
            seed=args.seed,
            device_name=args.device,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        payload.update(
            {
                "parameter_count": state.parameter_count,
                "model_fingerprint": state.model_fingerprint,
                "environment": environment_metadata(torch),
            }
        )
        failure_stage = "warmup_or_measurement"
        attention_context = patched_attention_ranges() if args.annotate_attention else nullcontext()
        with attention_context:
            payload.update(
                benchmark_state(
                    state,
                    warmup=args.warmup,
                    steps=args.steps,
                    stage_callback=lambda stage: payload.update(failure_stage=stage),
                )
            )
        payload["status"] = "success"
        payload["failure_stage"] = None
    except Exception as exc:
        payload.update(
            {
                "status": classify_error(exc),
                "failure_stage": payload.get("failure_stage") or failure_stage,
                "error_type": exc.__class__.__name__,
                "error_summary": safe_error_summary(exc),
                "environment": environment_metadata(torch),
            }
        )
        raise
    finally:
        payload["finished_at"] = utc_now()
        write_json(args.output, payload)
    return payload


def main() -> int:
    args = parse_args()
    run_benchmark(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
