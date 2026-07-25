from __future__ import annotations

import argparse
import json
import shlex
import statistics
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import torch

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
    """Parse and validate command-line arguments."""

    parser = argparse.ArgumentParser(
        description="A2-P end-to-end Transformer benchmark"
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
        default=512,
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=10_000,
    )
    parser.add_argument(
        "--mode",
        choices=[
            "forward",
            "forward_backward",
            "train_step",
        ],
        default="forward",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=10,
    )
    parser.add_argument(
        "--dtype",
        choices=["fp32"],
        default="fp32",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-3,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/benchmark/result.json"),
    )

    args = parser.parse_args()

    if args.batch_size <= 0:
        parser.error("--batch-size must be positive")

    if args.context_length <= 0:
        parser.error("--context-length must be positive")

    if args.vocab_size <= 0:
        parser.error("--vocab-size must be positive")

    if args.warmup < 0:
        parser.error("--warmup cannot be negative")

    if args.steps < 2:
        parser.error("--steps must be at least 2")

    if args.learning_rate <= 0:
        parser.error("--learning-rate must be positive")

    return args


def set_seed(seed: int) -> None:
    """Set the CPU and CUDA random seeds."""

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_model(args: argparse.Namespace) -> BasicsTransformerLM:
    """Create the requested Transformer model."""

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


def create_random_batch(
    args: argparse.Namespace,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Create random input token IDs and language-model targets."""

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


def main() -> None:
    args = parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for A2-P benchmarking")

    set_seed(args.seed)

    # 模型、数据和 optimizer 都在计时开始前创建。
    model = build_model(args)
    input_ids, targets = create_random_batch(args)

    optimizer: AdamW | None = None

    if args.mode == "train_step":
        optimizer = AdamW(
            model.parameters(),
            lr=args.learning_rate,
        )

    if args.mode == "forward":
        model.eval()
    else:
        model.train()

    def run_step() -> None:
        """Run exactly one step of the requested benchmark mode."""

        if args.mode == "forward":
            # forward-only 不需要构建 autograd 计算图。
            with torch.inference_mode():
                model(input_ids)
            return

        # forward_backward 和 train_step 每一步都清理梯度，
        # 防止梯度跨 measurement step 累积。
        model.zero_grad(set_to_none=True)

        logits = model(input_ids)
        loss = cross_entropy(logits, targets)
        loss.backward()

        if args.mode == "train_step":
            assert optimizer is not None
            optimizer.step()

    # Warm-up 阶段不计入正式测量。
    for _ in range(args.warmup):
        run_step()
        torch.cuda.synchronize()

    timings_ms: list[float] = []

    # 正式 measurement 阶段。
    for _ in range(args.steps):
        # 确保前面没有尚未完成的 CUDA 工作。
        torch.cuda.synchronize()

        start = time.perf_counter()

        run_step()

        # 等待当前 step 的所有 CUDA kernels 完成。
        torch.cuda.synchronize()

        elapsed_seconds = time.perf_counter() - start
        elapsed_ms = elapsed_seconds * 1000

        timings_ms.append(elapsed_ms)

    mean_ms = statistics.mean(timings_ms)
    sample_std_ms = statistics.stdev(timings_ms)
    cv = sample_std_ms / mean_ms

    result = {
        "metadata": {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "command": shlex.join(
                [".venv/bin/python", *sys.argv]
            ),
            "model_size": args.model_size,
            "model_config": MODEL_CONFIGS[args.model_size],
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "vocab_size": args.vocab_size,
            "mode": args.mode,
            "warmup": args.warmup,
            "steps": args.steps,
            "dtype": args.dtype,
            "seed": args.seed,
            "learning_rate": (
                args.learning_rate
                if args.mode == "train_step"
                else None
            ),
            "pytorch_version": torch.__version__,
            "cuda_runtime": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(0),
        },
        "raw_timings_ms": timings_ms,
        "summary": {
            "mean_ms": mean_ms,
            "sample_std_ms": sample_std_ms,
            "cv": cv,
        },
    }

    args.output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    args.output.write_text(
        json.dumps(result, indent=2),
        encoding="utf-8",
    )

    formatted_timings = ", ".join(
        f"{timing:.3f}" for timing in timings_ms
    )

    print(f"mode: {args.mode}")
    print(f"raw timings: [{formatted_timings}] ms")
    print(f"mean: {mean_ms:.3f} ms")
    print(f"sample std: {sample_std_ms:.3f} ms")
    print(f"CV: {cv:.4f} ({cv * 100:.2f}%)")
    print(f"saved: {args.output}")


if __name__ == "__main__":
    main()