"""Common CUDA measurement primitives used by all A2-P experiments."""

from __future__ import annotations

import gc
import json
import math
import platform
import random
import re
import statistics
import subprocess
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, ContextManager

import numpy as np
import torch
from torch.profiler import record_function

try:
    from .config import RunConfig, model_kwargs
except ImportError:  # Allows direct execution from the profiling directory.
    from config import RunConfig, model_kwargs


MIB = 2**20
OOM_ALLOCATION = re.compile(r"Tried to allocate ([0-9.]+) (KiB|MiB|GiB)")


@dataclass
class StepResult:
    """Outputs needed for statistics without retaining the full logits tensor."""

    loss: float | None
    stage_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]]


def require_cuda() -> torch.device:
    if not torch.cuda.is_available():
        raise RuntimeError("A2-P profiling requires an available CUDA device")
    return torch.device("cuda", 0)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _driver_version() -> str:
    try:
        completed = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=driver_version",
                "--format=csv,noheader",
            ],
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return completed.stdout.strip().splitlines()[0]


def public_environment() -> dict[str, Any]:
    """Return only hardware/software fields safe for a public report."""

    device = require_cuda()
    properties = torch.cuda.get_device_properties(device)
    return {
        "python": platform.python_version(),
        "pytorch": torch.__version__,
        "pytorch_compiled_cuda": torch.version.cuda,
        "driver": _driver_version(),
        "gpu_model": properties.name,
        "gpu_total_memory_mib": round(properties.total_memory / MIB, 2),
        "compute_capability": f"{properties.major}.{properties.minor}",
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "compute_profiler": f"torch.profiler {torch.__version__}",
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def build_model(config: RunConfig) -> torch.nn.Module:
    """Instantiate the fixed starter TransformerLM and move it to CUDA."""

    from cs336_basics.model import BasicsTransformerLM

    config.validate()
    seed_everything(config.seed)
    model = BasicsTransformerLM(
        **model_kwargs(config.model_size, config.context_length)
    )
    return model.to(device=require_cuda(), dtype=torch.float32)


def build_batch(config: RunConfig) -> tuple[torch.Tensor, torch.Tensor]:
    """Create fixed-shape synthetic token and label tensors outside timing."""

    device = require_cuda()
    generator = torch.Generator(device=device).manual_seed(config.seed + 1)
    shape = (config.batch_size, config.context_length)
    inputs = torch.randint(
        0,
        10_000,
        shape,
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    targets = torch.randint(
        0,
        10_000,
        shape,
        generator=generator,
        device=device,
        dtype=torch.long,
    )
    return inputs, targets


def build_optimizer(model: torch.nn.Module) -> torch.optim.Optimizer:
    from cs336_basics.optimizer import AdamW

    return AdamW(model.parameters(), lr=1e-3)


def autocast_context(dtype: str) -> ContextManager[Any]:
    if dtype == "fp32":
        return nullcontext()
    if dtype == "bf16":
        return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
    raise ValueError(f"unsupported dtype: {dtype}")


def _cuda_stage(
    name: str,
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
) -> tuple[torch.cuda.Event, torch.cuda.Event]:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    events[name] = (start, end)
    return start, end


def run_step(
    model: torch.nn.Module,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype: str,
    optimizer: torch.optim.Optimizer | None = None,
    capture_stage_events: bool = False,
) -> StepResult:
    """Run exactly one requested workload with explicit profiler stage ranges."""

    from cs336_basics.nn_utils import cross_entropy

    if mode not in {"forward", "forward_backward", "train_step"}:
        raise ValueError(f"unsupported mode: {mode}")
    if mode == "train_step" and optimizer is None:
        raise ValueError("train_step requires an optimizer")

    stage_events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]] = {}
    if mode != "forward":
        with record_function("zero_grad"):
            if optimizer is not None:
                optimizer.zero_grad(set_to_none=True)
            else:
                model.zero_grad(set_to_none=True)

    forward_events = (
        _cuda_stage("forward", stage_events) if capture_stage_events else None
    )
    with record_function("forward"):
        if forward_events:
            forward_events[0].record()
        grad_context = torch.no_grad() if mode == "forward" else nullcontext()
        with grad_context, autocast_context(dtype):
            logits = model(inputs)
            loss_tensor = (
                None if mode == "forward" else cross_entropy(logits, targets)
            )
        if forward_events:
            forward_events[1].record()

    if mode == "forward":
        del logits
        return StepResult(loss=None, stage_events=stage_events)

    backward_events = (
        _cuda_stage("backward", stage_events) if capture_stage_events else None
    )
    with record_function("backward"):
        if backward_events:
            backward_events[0].record()
        assert loss_tensor is not None
        loss_tensor.backward()
        if backward_events:
            backward_events[1].record()

    if mode == "train_step":
        optimizer_events = (
            _cuda_stage("optimizer", stage_events)
            if capture_stage_events
            else None
        )
        with record_function("optimizer"):
            if optimizer_events:
                optimizer_events[0].record()
            assert optimizer is not None
            optimizer.step()
            if optimizer_events:
                optimizer_events[1].record()

    loss = float(loss_tensor.detach().float().item())
    del logits, loss_tensor
    return StepResult(loss=loss, stage_events=stage_events)


def stage_elapsed_ms(
    events: dict[str, tuple[torch.cuda.Event, torch.cuda.Event]],
) -> dict[str, float]:
    """Resolve CUDA event pairs after the enclosing step was synchronized."""

    return {
        name: round(start.elapsed_time(end), 6)
        for name, (start, end) in events.items()
    }


def timing_statistics(timings_ms: list[float]) -> dict[str, float]:
    if not timings_ms:
        raise ValueError("at least one timing is required")
    mean = statistics.fmean(timings_ms)
    sample_std = statistics.stdev(timings_ms) if len(timings_ms) > 1 else 0.0
    return {
        "mean_ms": mean,
        "sample_std_ms": sample_std,
        "cv": sample_std / mean if mean else math.nan,
        "min_ms": min(timings_ms),
        "max_ms": max(timings_ms),
    }


def memory_stats_mib() -> dict[str, float]:
    stats = torch.cuda.memory_stats()
    return {
        "active_current_mib": stats["active_bytes.all.current"] / MIB,
        "active_peak_mib": stats["active_bytes.all.peak"] / MIB,
        "allocated_current_mib": torch.cuda.memory_allocated() / MIB,
        "allocated_peak_mib": torch.cuda.max_memory_allocated() / MIB,
        "reserved_current_mib": torch.cuda.memory_reserved() / MIB,
        "reserved_peak_mib": torch.cuda.max_memory_reserved() / MIB,
    }


def sanitize_oom(error: BaseException) -> dict[str, str | None]:
    """Extract useful OOM evidence without copying paths or process details."""

    message = str(error)
    match = OOM_ALLOCATION.search(message)
    attempted = f"{match.group(1)} {match.group(2)}" if match else None
    return {
        "error_type": type(error).__name__,
        "attempted_allocation": attempted,
    }


def release_cuda() -> None:
    """Drop references and release cache between independent configurations."""

    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.synchronize()
