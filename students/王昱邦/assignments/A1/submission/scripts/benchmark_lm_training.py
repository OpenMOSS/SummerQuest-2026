from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optimizer import AdamW, clip_gradients


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Benchmark end-to-end training steps for the A1 Transformer."
    )
    parser.add_argument("--train-data-path", type=Path, required=True)
    parser.add_argument("--dataset-dtype", default="uint16")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=10)
    parser.add_argument("--seed", type=int, default=336)
    parser.add_argument(
        "--matmul-precision",
        choices=("highest", "high", "medium"),
        default="highest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.train_data_path.is_file():
        raise FileNotFoundError(args.train_data_path)
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("warmup_steps must be non-negative and measurement_steps positive.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)
    device = torch.device("cuda")
    dataset = np.memmap(
        args.train_data_path,
        dtype=np.dtype(args.dataset_dtype),
        mode="r",
    )

    model_config = {
        "vocab_size": 10_000,
        "context_length": 256,
        "d_model": 512,
        "num_layers": 4,
        "num_heads": 16,
        "d_ff": 1344,
        "rope_theta": 10_000.0,
    }
    model = TransformerLM(
        **model_config,
        device=device,
        dtype=torch.float32,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=3e-4,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
    )
    parameter_count = sum(parameter.numel() for parameter in model.parameters())

    def run_step() -> tuple[float, float]:
        inputs, targets = get_batch(
            dataset=dataset,
            batch_size=args.batch_size,
            context_length=model_config["context_length"],
            device=device,
        )
        optimizer.zero_grad()
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        gradient_norm = clip_gradients(model.parameters(), max_l2_norm=1.0)
        optimizer.step()
        return loss.item(), gradient_norm.item()

    for _ in range(args.warmup_steps):
        run_step()
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    step_seconds: list[float] = []
    losses: list[float] = []
    gradient_norms: list[float] = []
    for _ in range(args.measurement_steps):
        start_time = time.perf_counter()
        loss, gradient_norm = run_step()
        torch.cuda.synchronize()
        step_seconds.append(time.perf_counter() - start_time)
        losses.append(loss)
        gradient_norms.append(gradient_norm)

    tokens_per_step = args.batch_size * model_config["context_length"]
    total_measurement_seconds = sum(step_seconds)
    result = {
        "device": torch.cuda.get_device_name(),
        "torch_version": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "matmul_precision": torch.get_float32_matmul_precision(),
        "model": model_config,
        "parameter_count": parameter_count,
        "dtype": "float32",
        "batch_size": args.batch_size,
        "tokens_per_step": tokens_per_step,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "mean_step_seconds": statistics.mean(step_seconds),
        "median_step_seconds": statistics.median(step_seconds),
        "min_step_seconds": min(step_seconds),
        "max_step_seconds": max(step_seconds),
        "tokens_per_second": (
            tokens_per_step * args.measurement_steps / total_measurement_seconds
        ),
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "final_loss": losses[-1],
        "final_gradient_norm": gradient_norms[-1],
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
