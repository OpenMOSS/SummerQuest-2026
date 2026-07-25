from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import torch.profiler
from tqdm.auto import tqdm

from profiling.common import (
    BenchmarkRunConfig,
    autocast_context,
    build_model,
    config_metadata,
    ensure_parent_dir,
    make_optimizer,
    random_batch,
    require_cuda,
    seed_everything,
    summarize_timings,
    synchronize,
    timer,
    write_json,
)
from profiling.nvtx_ranges import install_attention_nvtx, nvtx_range, nvtx_range_in_domain


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark CS336 Transformer forward/backward/train step timings.")
    parser.add_argument("--model-size", choices=["small", "medium", "large", "xl", "10B"], required=True)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], required=True)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--torch-compile", action="store_true")
    parser.add_argument("--enable-nvtx", action="store_true")
    parser.add_argument("--torch-profiler-output", type=str, default=None)
    parser.add_argument("--torch-profiler-steps", type=int, default=3)
    parser.add_argument("--torch-profiler-record-shapes", action="store_true")
    parser.add_argument("--torch-profiler-profile-memory", action="store_true")
    parser.add_argument("--torch-profiler-with-stack", action="store_true")
    parser.add_argument("--disable-progress", action="store_true")
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()


def run_single_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype_name: str,
) -> dict[str, float]:
    metrics: dict[str, float] = {
        "forward_ms": 0.0,
        "backward_ms": 0.0,
        "optimizer_ms": 0.0,
        "total_ms": 0.0,
        "loss": 0.0,
    }

    optimizer.zero_grad(set_to_none=True)
    step_start = timer()

    forward_start = timer()
    with nvtx_range("forward"):
        with autocast_context(dtype_name):
            if mode == "forward":
                with torch.no_grad():
                    _ = model(inputs)
            else:
                logits = model(inputs)
                loss = torch.nn.functional.cross_entropy(
                    logits.reshape(-1, logits.size(-1)),
                    targets.reshape(-1),
                )
                metrics["loss"] = float(loss.detach().item())
    synchronize()
    forward_end = timer()
    metrics["forward_ms"] = (forward_end - forward_start) * 1000.0

    if mode in {"forward_backward", "train_step"}:
        backward_start = timer()
        with nvtx_range("backward"):
            loss.backward()
        synchronize()
        backward_end = timer()
        metrics["backward_ms"] = (backward_end - backward_start) * 1000.0

    if mode == "train_step":
        optimizer_start = timer()
        with nvtx_range("optimizer"):
            optimizer.step()
        synchronize()
        optimizer_end = timer()
        metrics["optimizer_ms"] = (optimizer_end - optimizer_start) * 1000.0

    metrics["total_ms"] = (timer() - step_start) * 1000.0
    return metrics


def export_torch_profiler_trace(
    *,
    output_path: str,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    inputs: torch.Tensor,
    targets: torch.Tensor,
    mode: str,
    dtype_name: str,
    steps: int,
    record_shapes: bool,
    profile_memory: bool,
    with_stack: bool,
) -> dict[str, Any]:
    ensure_parent_dir(output_path)
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
        record_shapes=record_shapes,
        profile_memory=profile_memory,
        with_stack=with_stack,
    ) as prof:
        for _ in range(steps):
            run_single_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                mode=mode,
                dtype_name=dtype_name,
            )
            prof.step()

    prof.export_chrome_trace(output_path)
    return {
        "trace_path": output_path,
        "steps": steps,
        "record_shapes": record_shapes,
        "profile_memory": profile_memory,
        "with_stack": with_stack,
    }


def main() -> None:
    args = parse_args()
    require_cuda()

    config = BenchmarkRunConfig(
        model_size=args.model_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        warmup=args.warmup,
        steps=args.steps,
        dtype=args.dtype,
        seed=args.seed,
        vocab_size=args.vocab_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        torch_compile=args.torch_compile,
        enable_nvtx=args.enable_nvtx,
    )

    seed_everything(config.seed)
    if config.enable_nvtx:
        install_attention_nvtx()

    model = build_model(
        model_size=config.model_size,
        context_length=config.context_length,
        vocab_size=config.vocab_size,
        torch_compile=config.torch_compile,
    )
    optimizer = make_optimizer(
        model,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        betas=config.optimizer_betas,
        eps=config.optimizer_eps,
    )
    inputs, targets = random_batch(config.batch_size, config.context_length, config.vocab_size)

    if config.mode == "forward":
        model.eval()
    else:
        model.train()

    show_progress = not args.disable_progress
    with nvtx_range_in_domain("profile", "warmup"):
        for _ in tqdm(
            range(config.warmup),
            desc=f"warmup:{config.model_size}:{config.mode}:{config.dtype}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress or config.warmup == 0,
        ):
            run_single_step(
                model=model,
                optimizer=optimizer,
                inputs=inputs,
                targets=targets,
                mode=config.mode,
                dtype_name=config.dtype,
            )

    timing_records: list[dict[str, float]] = []
    with nvtx_range_in_domain("profile", "measure"):
        for _ in tqdm(
            range(config.steps),
            desc=f"measure:{config.model_size}:{config.mode}:{config.dtype}",
            leave=False,
            dynamic_ncols=True,
            disable=not show_progress or config.steps == 0,
        ):
            timing_records.append(
                run_single_step(
                    model=model,
                    optimizer=optimizer,
                    inputs=inputs,
                    targets=targets,
                    mode=config.mode,
                    dtype_name=config.dtype,
                )
            )

    per_metric = {
        "total_ms": [record["total_ms"] for record in timing_records],
        "forward_ms": [record["forward_ms"] for record in timing_records],
        "backward_ms": [record["backward_ms"] for record in timing_records if record["backward_ms"] > 0],
        "optimizer_ms": [record["optimizer_ms"] for record in timing_records if record["optimizer_ms"] > 0],
        "loss": [record["loss"] for record in timing_records if record["loss"] > 0],
    }

    profiler_trace = None
    if args.torch_profiler_output:
        profiler_trace = export_torch_profiler_trace(
            output_path=args.torch_profiler_output,
            model=model,
            optimizer=optimizer,
            inputs=inputs,
            targets=targets,
            mode=config.mode,
            dtype_name=config.dtype,
            steps=args.torch_profiler_steps,
            record_shapes=args.torch_profiler_record_shapes,
            profile_memory=args.torch_profiler_profile_memory,
            with_stack=args.torch_profiler_with_stack,
        )

    result = config_metadata(config)
    result.update(
        {
            "artifact_type": "benchmark",
            "output_path": str(Path(args.output)),
            "timing_records": timing_records,
            "summaries": {metric: summarize_timings(values) for metric, values in per_metric.items()},
            "torch_profiler": profiler_trace,
        }
    )

    write_json(args.output, result)
    print(f"Saved benchmark results to {args.output}")
    print(f"Mean total step time: {result['summaries']['total_ms']['mean_ms']:.3f} ms")


if __name__ == "__main__":
    main()
