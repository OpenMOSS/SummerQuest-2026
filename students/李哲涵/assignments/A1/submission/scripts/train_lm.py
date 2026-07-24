from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from _project_api import (
    build_model,
    build_optimizer,
    get_batch_fn as resolve_get_batch_fn,
    get_cross_entropy_fn as resolve_cross_entropy_fn,
    get_gradient_clipping_fn as resolve_gradient_clipping_fn,
    get_load_checkpoint_fn as resolve_load_checkpoint_fn,
    get_lr_schedule_fn as resolve_lr_schedule_fn,
    get_save_checkpoint_fn as resolve_save_checkpoint_fn,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a Transformer language model.")
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--theta", type=float, default=10_000.0)
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument(
        "--norm-position",
        choices=("pre", "post", "none"),
        default="pre",
    )
    parser.add_argument("--no-final-norm", action="store_true")
    parser.add_argument("--ffn-type", choices=("swiglu", "silu"), default="swiglu")
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--max-steps", type=int, required=True)
    parser.add_argument("--lr", type=float, required=True)
    parser.add_argument("--min-lr", type=float, required=True)
    parser.add_argument("--warmup-steps", type=int, required=True)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=100)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--resume", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_int_fields = (
        "vocab_size",
        "context_length",
        "d_model",
        "d_ff",
        "num_layers",
        "num_heads",
        "batch_size",
        "max_steps",
        "log_every",
        "eval_every",
        "eval_batches",
        "save_every",
    )
    for field in positive_int_fields:
        if getattr(args, field) <= 0:
            raise ValueError(f"--{field.replace('_', '-')} 必须大于 0")
    if args.d_model % args.num_heads != 0:
        raise ValueError("--d-model 必须能被 --num-heads 整除")
    if args.warmup_steps < 0:
        raise ValueError("--warmup-steps 不能为负数")
    if not (0.0 <= args.min_lr <= args.lr):
        raise ValueError("需要满足 0 <= min_lr <= lr")
    if args.grad_clip <= 0:
        raise ValueError("--grad-clip 必须大于 0")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def flatten_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    cross_entropy_fn: Any,
) -> torch.Tensor:
    return cross_entropy_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))


@torch.no_grad()
def evaluate(
    model: torch.nn.Module,
    valid_data: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
    get_batch_fn: Any,
    cross_entropy_fn: Any,
) -> float:
    model.eval()
    losses: list[float] = []
    for _ in range(num_batches):
        inputs, targets = get_batch_fn(valid_data, batch_size, context_length, device)
        logits = model(inputs)
        loss = flatten_loss(logits, targets, cross_entropy_fn)
        losses.append(float(loss.item()))
    model.train()
    return float(sum(losses) / len(losses))


def append_metric(path: Path, metric: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(metric, ensure_ascii=False) + "\n")


def save_config(path: Path, config: dict[str, Any]) -> None:
    serializable = {
        key: str(value) if isinstance(value, Path) else value
        for key, value in config.items()
    }
    with path.open("w", encoding="utf-8") as file:
        json.dump(serializable, file, ensure_ascii=False, indent=2)
        file.write("\n")


def main() -> None:
    args = parse_args()
    validate_args(args)
    for path in (args.train_data, args.valid_data):
        if not path.is_file():
            raise FileNotFoundError(path)

    args.run_dir.mkdir(parents=True, exist_ok=True)
    config = vars(args).copy()
    save_config(args.run_dir / "config.json", config)
    metrics_path = args.run_dir / "metrics.jsonl"

    set_seed(args.seed)
    device = torch.device(args.device)
    train_data = np.load(args.train_data, mmap_mode="r", allow_pickle=False)
    valid_data = np.load(args.valid_data, mmap_mode="r", allow_pickle=False)
    if train_data.ndim != 1 or valid_data.ndim != 1:
        raise ValueError("train/valid .npy 必须是一维 token id 数组")
    if len(train_data) <= args.context_length or len(valid_data) <= args.context_length:
        raise ValueError("数据长度必须大于 context_length")

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)

    get_batch_fn = resolve_get_batch_fn()
    cross_entropy_fn = resolve_cross_entropy_fn()
    clip_grad_fn = resolve_gradient_clipping_fn()
    lr_schedule_fn = resolve_lr_schedule_fn()
    save_checkpoint_fn = resolve_save_checkpoint_fn()
    load_checkpoint_fn = resolve_load_checkpoint_fn()

    start_step = 0
    if args.resume is not None:
        if not args.resume.is_file():
            raise FileNotFoundError(args.resume)
        start_step = int(load_checkpoint_fn(args.resume, model, optimizer))
        print(f"resumed from step {start_step}: {args.resume}")
    if start_step >= args.max_steps:
        raise ValueError(
            f"checkpoint step={start_step} 已经不小于 max_steps={args.max_steps}"
        )

    optimizer.zero_grad(set_to_none=True)
    model.train()
    wall_start = time.perf_counter()

    for step in range(start_step + 1, args.max_steps + 1):
        step_start = time.perf_counter()
        lr = float(
            lr_schedule_fn(
                step - 1,
                args.lr,
                args.min_lr,
                args.warmup_steps,
                args.max_steps,
            )
        )
        for group in optimizer.param_groups:
            group["lr"] = lr

        inputs, targets = get_batch_fn(
            train_data,
            args.batch_size,
            args.context_length,
            args.device,
        )
        logits = model(inputs)
        loss = flatten_loss(logits, targets, cross_entropy_fn)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"step {step}: loss is not finite: {loss.item()}")

        loss.backward()
        clip_grad_fn(model.parameters(), args.grad_clip)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)

        if step % args.log_every == 0 or step == 1 or step == args.max_steps:
            metric = {
                "step": step,
                "split": "train",
                "loss": float(loss.item()),
                "lr": lr,
                "step_seconds": time.perf_counter() - step_start,
                "elapsed_seconds": time.perf_counter() - wall_start,
            }
            append_metric(metrics_path, metric)
            print(
                f"step={step:>6d} train_loss={metric['loss']:.6f} "
                f"lr={lr:.3e} step_time={metric['step_seconds']:.3f}s"
            )

        if step % args.eval_every == 0 or step == args.max_steps:
            valid_loss = evaluate(
                model=model,
                valid_data=valid_data,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=args.device,
                num_batches=args.eval_batches,
                get_batch_fn=get_batch_fn,
                cross_entropy_fn=cross_entropy_fn,
            )
            metric = {
                "step": step,
                "split": "valid",
                "loss": valid_loss,
                "perplexity": math.exp(valid_loss) if valid_loss < 709 else math.inf,
                "elapsed_seconds": time.perf_counter() - wall_start,
            }
            append_metric(metrics_path, metric)
            print(
                f"step={step:>6d} valid_loss={valid_loss:.6f} "
                f"perplexity={metric['perplexity']:.3f}"
            )

        if step % args.save_every == 0 or step == args.max_steps:
            checkpoint_path = args.run_dir / f"step_{step}.pt"
            save_checkpoint_fn(model, optimizer, step, checkpoint_path)
            print(f"saved checkpoint: {checkpoint_path}")


if __name__ == "__main__":
    main()
