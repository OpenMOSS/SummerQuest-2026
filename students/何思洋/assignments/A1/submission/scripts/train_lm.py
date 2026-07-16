from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    cross_entropy,
    get_batch,
    get_lr_cosine_schedule,
    gradient_clipping,
    load_checkpoint,
    save_checkpoint,
)


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


@torch.no_grad()
def estimate_val_loss(
    model: torch.nn.Module,
    val_tokens: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(val_tokens, batch_size, context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(sum(losses) / len(losses))


def append_jsonl(path: Path, record: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer LM.")
    parser.add_argument("--train-tokens", type=Path, required=True)
    parser.add_argument("--val-tokens", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--run-name", default="train_lm")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--norm-mode", choices=["pre", "post", "none"], default="pre")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-activation", choices=["swiglu", "linear-gate"], default="swiglu")

    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max-lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=5e-5)
    parser.add_argument("--warmup-iters", type=int, default=500)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-interval", type=int, default=10)
    parser.add_argument("--eval-interval", type=int, default=500)
    parser.add_argument("--eval-batches", type=int, default=20)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)
    parser.add_argument("--resume", type=Path, default=None)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = choose_device(args.device)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = args.out_dir / "checkpoints"
    logs_dir = args.out_dir / "logs"
    log_path = logs_dir / f"{args.run_name}.jsonl"
    summary_path = logs_dir / f"{args.run_name}_summary.json"

    train_tokens = np.load(args.train_tokens, mmap_mode="r")
    val_tokens = np.load(args.val_tokens, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        norm_mode=args.norm_mode,
        use_rope=not args.no_rope,
        ffn_activation=args.ffn_activation,
    ).to(device)
    optimizer = AdamW(model.parameters(), lr=args.max_lr, weight_decay=args.weight_decay)

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    config = {key: str(value) if isinstance(value, Path) else value for key, value in vars(args).items()}
    config["device"] = device
    config["num_parameters"] = sum(parameter.numel() for parameter in model.parameters())
    config["train_tokens_count"] = int(train_tokens.shape[0])
    config["val_tokens_count"] = int(val_tokens.shape[0])

    start_time = time.perf_counter()
    last_train_loss = None
    last_val_loss = None

    model.train()
    for step in range(start_step, args.steps):
        lr = get_lr_cosine_schedule(step, args.max_lr, args.min_lr, args.warmup_iters, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch(train_tokens, args.batch_size, args.context_length, device)
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        completed_step = step + 1
        last_train_loss = float(loss.item())
        should_eval = completed_step == 1 or completed_step % args.eval_interval == 0 or completed_step == args.steps
        if should_eval:
            last_val_loss = estimate_val_loss(
                model,
                val_tokens,
                batch_size=args.batch_size,
                context_length=args.context_length,
                device=device,
                num_batches=args.eval_batches,
            )

        if completed_step == 1 or completed_step % args.log_interval == 0 or should_eval:
            record = {
                "step": completed_step,
                "wall_clock_sec": time.perf_counter() - start_time,
                "train_loss": last_train_loss,
                "lr": lr,
                "processed_tokens": completed_step * args.batch_size * args.context_length,
            }
            if should_eval:
                record["val_loss"] = last_val_loss
            append_jsonl(log_path, record)
            print(json.dumps(record), flush=True)

        if completed_step % args.checkpoint_interval == 0 or completed_step == args.steps:
            checkpoints_dir.mkdir(parents=True, exist_ok=True)
            save_checkpoint(model, optimizer, completed_step, checkpoints_dir / f"{args.run_name}_step{completed_step}.pt")

    total_time = time.perf_counter() - start_time
    summary = {
        **config,
        "final_step": args.steps,
        "final_train_loss": last_train_loss,
        "final_val_loss": last_val_loss,
        "total_train_wall_clock_sec": total_time,
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
