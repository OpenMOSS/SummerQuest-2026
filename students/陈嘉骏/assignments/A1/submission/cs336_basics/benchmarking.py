from __future__ import annotations

import gc
import json
import os
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy.typing as npt
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW, clip_gradients, cross_entropy, get_batch


@dataclass(frozen=True)
class BatchSizeBenchmarkResult:
    batch_size: int
    status: str
    warmup_steps: int
    measured_steps: int
    tokens_per_step: int
    total_measured_tokens: int
    data_seconds_per_step: float | None
    compute_seconds_per_step: float | None
    end_to_end_seconds_per_step: float | None
    compute_tokens_per_second: float | None
    end_to_end_tokens_per_second: float | None
    peak_memory_allocated_bytes: int | None
    peak_memory_reserved_bytes: int | None
    final_loss: float | None
    error: str | None = None


@dataclass(frozen=True)
class BatchSizeBenchmarkSummary:
    device: str
    gpu_name: str
    gpu_total_memory_bytes: int
    torch_version: str
    torch_cuda_version: str | None
    model_config: dict[str, Any]
    context_length: int
    learning_rate: float
    max_grad_norm: float
    results: list[BatchSizeBenchmarkResult]


def benchmark_batch_sizes(
    model_config: dict[str, Any],
    dataset: npt.NDArray[Any],
    batch_sizes: list[int],
    device: torch.device | str,
    warmup_steps: int = 5,
    measured_steps: int = 20,
    learning_rate: float = 3e-4,
    max_grad_norm: float = 1.0,
    seed: int = 42,
    on_result: Callable[[BatchSizeBenchmarkResult], None] | None = None,
) -> BatchSizeBenchmarkSummary:
    """Benchmark full training steps over a batch-size sweep on one CUDA device."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda":
        raise ValueError("Batch-size memory benchmarking requires a CUDA device.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available to PyTorch.")
    if not batch_sizes or any(batch_size <= 0 for batch_size in batch_sizes):
        raise ValueError("batch_sizes must contain positive integers.")
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("warmup_steps and measured_steps must be positive.")
    if learning_rate <= 0 or max_grad_norm <= 0:
        raise ValueError("learning_rate and max_grad_norm must be positive.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    context_length = int(model_config["context_length"])
    results: list[BatchSizeBenchmarkResult] = []

    for batch_size in batch_sizes:
        result = _benchmark_one_batch_size(
            model_config=model_config,
            dataset=dataset,
            batch_size=batch_size,
            context_length=context_length,
            device=resolved_device,
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
        )
        results.append(result)
        if on_result is not None:
            on_result(result)

    properties = torch.cuda.get_device_properties(resolved_device)
    return BatchSizeBenchmarkSummary(
        device=str(resolved_device),
        gpu_name=properties.name,
        gpu_total_memory_bytes=properties.total_memory,
        torch_version=torch.__version__,
        torch_cuda_version=torch.version.cuda,
        model_config=dict(model_config),
        context_length=context_length,
        learning_rate=learning_rate,
        max_grad_norm=max_grad_norm,
        results=results,
    )


def save_batch_size_benchmark(
    summary: BatchSizeBenchmarkSummary,
    output_path: str | os.PathLike[str],
) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary_output = output.with_name(output.name + ".tmp")
    try:
        with open(temporary_output, "w", encoding="utf-8") as f:
            json.dump(asdict(summary), f, ensure_ascii=False, indent=2)
            f.write("\n")
        os.replace(temporary_output, output)
    except BaseException:
        temporary_output.unlink(missing_ok=True)
        raise


def _benchmark_one_batch_size(
    model_config: dict[str, Any],
    dataset: npt.NDArray[Any],
    batch_size: int,
    context_length: int,
    device: torch.device,
    warmup_steps: int,
    measured_steps: int,
    learning_rate: float,
    max_grad_norm: float,
) -> BatchSizeBenchmarkResult:
    model: TransformerLM | None = None
    optimizer: AdamW | None = None
    torch.cuda.empty_cache()
    gc.collect()

    try:
        model = TransformerLM(**model_config, device=device)
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.1,
        )
        model.train()

        for _ in range(warmup_steps):
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            clip_gradients(model.parameters(), max_grad_norm)
            optimizer.step()
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)

        total_data_seconds = 0.0
        total_compute_seconds = 0.0
        final_loss = float("nan")
        for _ in range(measured_steps):
            data_start = time.perf_counter()
            inputs, targets = get_batch(dataset, batch_size, context_length, device)
            torch.cuda.synchronize(device)
            compute_start = time.perf_counter()
            total_data_seconds += compute_start - data_start

            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = cross_entropy(logits, targets)
            if not bool(torch.isfinite(loss)):
                raise RuntimeError(f"Non-finite loss for batch size {batch_size}: {float(loss)}")
            loss.backward()
            clip_gradients(model.parameters(), max_grad_norm)
            optimizer.step()
            torch.cuda.synchronize(device)
            total_compute_seconds += time.perf_counter() - compute_start
            final_loss = float(loss.detach())

        tokens_per_step = batch_size * context_length
        total_tokens = measured_steps * tokens_per_step
        data_seconds_per_step = total_data_seconds / measured_steps
        compute_seconds_per_step = total_compute_seconds / measured_steps
        end_to_end_seconds_per_step = data_seconds_per_step + compute_seconds_per_step
        return BatchSizeBenchmarkResult(
            batch_size=batch_size,
            status="ok",
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            tokens_per_step=tokens_per_step,
            total_measured_tokens=total_tokens,
            data_seconds_per_step=data_seconds_per_step,
            compute_seconds_per_step=compute_seconds_per_step,
            end_to_end_seconds_per_step=end_to_end_seconds_per_step,
            compute_tokens_per_second=total_tokens / total_compute_seconds,
            end_to_end_tokens_per_second=total_tokens / (total_data_seconds + total_compute_seconds),
            peak_memory_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_memory_reserved_bytes=torch.cuda.max_memory_reserved(device),
            final_loss=final_loss,
        )
    except (torch.OutOfMemoryError, RuntimeError) as error:
        if not isinstance(error, torch.OutOfMemoryError) and "out of memory" not in str(error).lower():
            raise
        return BatchSizeBenchmarkResult(
            batch_size=batch_size,
            status="oom",
            warmup_steps=warmup_steps,
            measured_steps=measured_steps,
            tokens_per_step=batch_size * context_length,
            total_measured_tokens=0,
            data_seconds_per_step=None,
            compute_seconds_per_step=None,
            end_to_end_seconds_per_step=None,
            compute_tokens_per_second=None,
            end_to_end_tokens_per_second=None,
            peak_memory_allocated_bytes=None,
            peak_memory_reserved_bytes=None,
            final_loss=None,
            error=str(error).splitlines()[0],
        )
    finally:
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        del optimizer
        del model
        gc.collect()
        torch.cuda.empty_cache()
