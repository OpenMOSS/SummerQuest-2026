from __future__ import annotations

import argparse
import csv
import json
import math
import shlex
import sys
from datetime import datetime, timezone
from pathlib import Path

import torch
from einops import einsum
from torch.profiler import (
    ProfilerActivity,
    profile,
    record_function,
    schedule,
    tensorboard_trace_handler,
)

import cs336_basics.model as model_impl
from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


MODEL_CONFIGS = {
    "small": {
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
    },
    "medium": {
        "d_model": 1024,
        "d_ff": 4096,
        "num_layers": 24,
        "num_heads": 16,
    },
    "large": {
        "d_model": 1280,
        "d_ff": 5120,
        "num_layers": 36,
        "num_heads": 20,
    },
    "xl": {
        "d_model": 2560,
        "d_ff": 10240,
        "num_layers": 32,
        "num_heads": 32,
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile one stable Transformer train_step"
    )

    parser.add_argument(
        "--model-size",
        choices=MODEL_CONFIGS,
        default="small",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--context-length",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results/profile/smoke_small_c256"),
    )
    parser.add_argument(
        "--run-name",
        type=str,
        default="smoke_small_c256",
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.context_length <= 0:
        parser.error("--context-length must be positive")

    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")

    if args.warmup < 1:
        parser.error("--warmup must be at least 1")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    return args


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> BasicsTransformerLM:
    config = MODEL_CONFIGS[args.model_size]

    model = BasicsTransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=config["d_model"],
        d_ff=config["d_ff"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        rope_theta=10_000.0,
    )

    return model.to(device="cuda", dtype=torch.float32)


def make_batch(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    input_ids = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device="cuda",
        dtype=torch.long,
    )

    targets = torch.randint(
        low=0,
        high=args.vocab_size,
        size=(args.batch_size, args.context_length),
        device="cuda",
        dtype=torch.long,
    )

    return input_ids, targets


def install_attention_profile_markers():
    """
    Add profiler markers around the existing scaled dot-product attention.

    The model calls cs336_basics.model.scaled_dot_product_attention
    from inside CausalMultiHeadSelfAttention.forward(). Replacing this
    module-level function lets us add profiling ranges without modifying
    the assignment model implementation.
    """

    original_attention = model_impl.scaled_dot_product_attention

    def profiled_scaled_dot_product_attention(
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        scale = 1.0 / math.sqrt(K.shape[-1])

        with record_function("attention/scores"):
            attention_scores = (
                einsum(
                    Q,
                    K,
                    "... query d_k, ... key d_k -> ... query key",
                )
                * scale
            )

            if mask is not None:
                attention_scores = torch.where(
                    mask,
                    attention_scores,
                    float("-inf"),
                )

        with record_function("attention/softmax"):
            attention_weights = model_impl.softmax(
                attention_scores,
                dim=-1,
            )

        with record_function("attention/value"):
            attention_output = einsum(
                attention_weights,
                V,
                "... query key, ... key d_v -> ... query d_v",
            )

        return attention_output

    model_impl.scaled_dot_product_attention = (
        profiled_scaled_dot_product_attention
    )

    return original_attention


def run_train_step(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)

    with record_function("forward"):
        logits = model(input_ids)

    with record_function("loss"):
        loss = cross_entropy(logits, targets)

    with record_function("backward"):
        loss.backward()

    with record_function("optimizer"):
        optimizer.step()

    return loss.detach()


def get_event_value(
    event,
    *field_names: str,
) -> float:
    """
    PyTorch has used both cuda_time_total and device_time_total
    names across versions. Support either one.
    """

    for field_name in field_names:
        value = getattr(event, field_name, None)
        if value is not None:
            return float(value)

    return 0.0


def save_operator_summary(
    profiler,
    output_path: Path,
) -> None:
    events = list(profiler.key_averages())

    events.sort(
        key=lambda event: get_event_value(
            event,
            "cuda_time_total",
            "device_time_total",
        ),
        reverse=True,
    )

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "name",
                "calls",
                "cpu_time_total_ms",
                "self_cpu_time_total_ms",
                "cuda_time_total_ms",
                "self_cuda_time_total_ms",
            ],
        )
        writer.writeheader()

        for event in events:
            writer.writerow(
                {
                    "name": event.key,
                    "calls": event.count,
                    "cpu_time_total_ms": (
                        get_event_value(
                            event,
                            "cpu_time_total",
                        )
                        / 1000.0
                    ),
                    "self_cpu_time_total_ms": (
                        get_event_value(
                            event,
                            "self_cpu_time_total",
                        )
                        / 1000.0
                    ),
                    "cuda_time_total_ms": (
                        get_event_value(
                            event,
                            "cuda_time_total",
                            "device_time_total",
                        )
                        / 1000.0
                    ),
                    "self_cuda_time_total_ms": (
                        get_event_value(
                            event,
                            "self_cuda_time_total",
                            "self_device_time_total",
                        )
                        / 1000.0
                    ),
                }
            )


def save_profiler_table(
    profiler,
    output_path: Path,
) -> None:
    try:
        table = profiler.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=40,
        )
    except (RuntimeError, ValueError):
        table = profiler.key_averages().table(
            sort_by="device_time_total",
            row_limit=40,
        )

    output_path.write_text(
        table,
        encoding="utf-8",
    )


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")

    set_seed(args.seed)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    model = build_model(args)
    input_ids, targets = make_batch(args)

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
    )

    model.train()

    original_attention = install_attention_profile_markers()

    try:
        activities = [
            ProfilerActivity.CPU,
            ProfilerActivity.CUDA,
        ]

        profiler_schedule = schedule(
            wait=args.warmup,
            warmup=0,
            active=1,
            repeat=1,
        )

        trace_handler = tensorboard_trace_handler(
            str(args.output_dir),
            worker_name=args.run_name,
        )

        with profile(
            activities=activities,
            schedule=profiler_schedule,
            on_trace_ready=trace_handler,
            record_shapes=True,
            profile_memory=False,
            with_stack=False,
        ) as profiler:
            for step_idx in range(args.warmup + 1):
                if step_idx < args.warmup:
                    phase_name = "profile/warmup"
                else:
                    phase_name = "profile/measure"

                with record_function(phase_name):
                    loss = run_train_step(
                        model=model,
                        optimizer=optimizer,
                        input_ids=input_ids,
                        targets=targets,
                    )

                torch.cuda.synchronize()
                profiler.step()

    finally:
        model_impl.scaled_dot_product_attention = original_attention

    summary_path = args.output_dir / "operator_summary.csv"
    table_path = args.output_dir / "profiler_table.txt"
    metadata_path = args.output_dir / "run_metadata.json"

    save_operator_summary(
        profiler,
        summary_path,
    )

    save_profiler_table(
        profiler,
        table_path,
    )

    trace_files = sorted(
        path.name
        for path in args.output_dir.iterdir()
        if path.name.endswith(".json")
        or path.name.endswith(".json.gz")
    )

    metadata = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "command": shlex.join(
            [".venv/bin/python", *sys.argv]
        ),
        "model_size": args.model_size,
        "model_config": MODEL_CONFIGS[args.model_size],
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "vocab_size": args.vocab_size,
        "mode": "train_step",
        "dtype": "fp32",
        "warmup": args.warmup,
        "profiled_steps": 1,
        "seed": args.seed,
        "learning_rate": args.learning_rate,
        "profiler": "torch.profiler",
        "activities": ["CPU", "CUDA"],
        "record_shapes": True,
        "profile_memory": False,
        "pytorch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "gpu_name": torch.cuda.get_device_name(0),
        "trace_files": trace_files,
        "loss_after_profile_step": float(loss.item()),
    }

    metadata_path.write_text(
        json.dumps(
            metadata,
            indent=2,
        ),
        encoding="utf-8",
    )

    print("profile completed")
    print(f"loss: {loss.item():.6f}")
    print(f"output directory: {args.output_dir}")
    print(f"operator summary: {summary_path}")
    print(f"profiler table: {table_path}")
    print(f"metadata: {metadata_path}")

    try:
        table = profiler.key_averages().table(
            sort_by="cuda_time_total",
            row_limit=20,
        )
    except (RuntimeError, ValueError):
        table = profiler.key_averages().table(
            sort_by="device_time_total",
            row_limit=20,
        )

    print(table)


if __name__ == "__main__":
    main()