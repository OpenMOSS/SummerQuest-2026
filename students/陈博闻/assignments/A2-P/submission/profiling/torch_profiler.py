from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import torch

A2_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(A2_ROOT))
sys.path.insert(0, str(A2_ROOT / "cs336-basics"))

from profiling.benchmark import (  # noqa: E402
    BenchmarkConfig,
    build_model,
    collect_environment,
    collect_memory,
    configure_device,
    install_attention_nvtx,
    make_batch,
    run_one_step,
    synchronize_if_needed,
)
from cs336_basics.optimizer import AdamW  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="torch.profiler trace for CS336 Assignment 2 profiling.")
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], default="train_step")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--dtype", choices=["fp32", "bf16", "fp16"], default="fp32")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--trace-output", type=Path, default=Path("results/torch/trace.json"))
    parser.add_argument("--summary-output", type=Path, default=Path("results/torch/summary.json"))
    parser.add_argument("--table-output", type=Path, default=Path("results/torch/operator_table.txt"))
    parser.add_argument("--record-shapes", action="store_true")
    parser.add_argument("--profile-memory", action="store_true")
    parser.add_argument("--with-stack", action="store_true")
    parser.add_argument("--no-nvtx", action="store_true")
    parser.add_argument("--no-attention-nvtx", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = configure_device(args.device)
    config = BenchmarkConfig(
        model_size=args.model_size,
        vocab_size=args.vocab_size,
        batch_size=args.batch_size,
        context_length=args.context_length,
        mode=args.mode,
        warmup=args.warmup,
        steps=args.steps,
        dtype=args.dtype,
        seed=args.seed,
        device=str(device),
        learning_rate=args.learning_rate,
        compile_model=False,
        nvtx=not args.no_nvtx,
    )

    torch.manual_seed(config.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(config.seed)
        torch.cuda.reset_peak_memory_stats(device)

    install_attention_nvtx(config.nvtx and not args.no_attention_nvtx)
    model = build_model(config, device)
    optimizer = AdamW(model.parameters(), lr=config.learning_rate)
    inputs, targets = make_batch(config, device)

    for _ in range(config.warmup):
        run_one_step(model, optimizer, inputs, targets, config, device)
    synchronize_if_needed(device)

    activities = [torch.profiler.ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(torch.profiler.ProfilerActivity.CUDA)

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    args.summary_output.parent.mkdir(parents=True, exist_ok=True)
    args.table_output.parent.mkdir(parents=True, exist_ok=True)

    with torch.profiler.profile(
        activities=activities,
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        acc_events=True,
    ) as prof:
        for step in range(config.steps):
            with torch.profiler.record_function(f"profile/measure_step_{step}"):
                run_one_step(model, optimizer, inputs, targets, config, device)
            synchronize_if_needed(device)
            prof.step()

    prof.export_chrome_trace(str(args.trace_output))
    table = prof.key_averages(group_by_input_shape=args.record_shapes).table(
        sort_by="self_cuda_time_total" if device.type == "cuda" else "self_cpu_time_total",
        row_limit=50,
    )
    args.table_output.write_text(table, encoding="utf-8")

    summary = {
        "config": asdict(config),
        "environment": collect_environment(device),
        "memory": collect_memory(device),
        "trace_output": str(args.trace_output),
        "table_output": str(args.table_output),
        "record_shapes": args.record_shapes,
        "profile_memory": args.profile_memory,
        "with_stack": args.with_stack,
    }
    args.summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
