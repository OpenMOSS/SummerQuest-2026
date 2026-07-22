"""Configurable training loop for the Transformer LM.

Reads token-id arrays via ``np.memmap`` so datasets never fully load into RAM.
Logs per-step JSONL records (step, wall_clock_sec, train_loss, lr, and val_loss
at eval steps) and writes a summary.json at the end. Supports checkpoint resume
and the four architecture ablations via CLI flags.

Example:
    python scripts/train.py --config configs/tinystories.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time

import numpy as np
import torch

from cs336_basics.data import get_batch, load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule


def load_memmap(path: str, dtype: str) -> np.ndarray:
    return np.load(path, mmap_mode="r") if path.endswith(".npy") else np.memmap(path, dtype=dtype, mode="r")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", default=None, help="Optional JSON file with any of the args below.")
    # data
    p.add_argument("--train-data", required=False)
    p.add_argument("--val-data", required=False)
    p.add_argument("--dtype", default="uint16")
    # model
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    # ablation switches
    p.add_argument("--norm", default="rmsnorm", choices=["rmsnorm", "none"])
    p.add_argument("--norm-position", default="pre", choices=["pre", "post"])
    p.add_argument("--use-rope", type=int, default=1)
    p.add_argument("--ffn", default="swiglu", choices=["swiglu", "silu"])
    # optimization
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--total-steps", type=int, default=10000)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--min-lr", type=float, default=3e-4)
    p.add_argument("--warmup-steps", type=int, default=200)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    p.add_argument("--grad-clip", type=float, default=1.0)
    # run control
    p.add_argument("--device", default="cuda")
    p.add_argument("--amp", type=int, default=1, help="bf16 autocast")
    p.add_argument("--compile", type=int, default=0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-batches", type=int, default=50)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--log-file", required=False)
    p.add_argument("--summary-file", required=False)
    p.add_argument("--ckpt", required=False, help="checkpoint output path")
    p.add_argument("--resume", default=None)
    p.add_argument("--max-runtime-sec", type=float, default=None)

    args = p.parse_args()
    if args.config:
        with open(args.config) as f:
            cfg = json.load(f)
        for k, v in cfg.items():
            setattr(args, k.replace("-", "_"), v)
    return args


@torch.no_grad()
def evaluate(model, val_data, args, device) -> float:
    model.eval()
    losses = []
    for _ in range(args.eval_batches):
        x, y = get_batch(val_data, args.batch_size, args.context_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    train_data = load_memmap(args.train_data, args.dtype)
    val_data = load_memmap(args.val_data, args.dtype) if args.val_data else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rope=bool(args.use_rope),
        norm=args.norm,
        norm_position=args.norm_position,
        ffn=args.ffn,
        device=device,
    ).to(device)
    if args.compile:
        model = torch.compile(model)

    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] parameters={n_params:,} device={device}", flush=True)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_step = 0
    if args.resume and os.path.exists(args.resume):
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"[train] resumed from {args.resume} at step {start_step}", flush=True)

    if args.log_file:
        os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    log_f = open(args.log_file, "a") if args.log_file else None

    def log(record: dict) -> None:
        if log_f:
            log_f.write(json.dumps(record) + "\n")
            log_f.flush()

    wall_start = time.time()
    best_val = math.inf
    final_val = None
    diverged = False
    model.train()

    for step in range(start_step, args.total_steps):
        lr = get_lr_cosine_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.total_steps)
        for g in optimizer.param_groups:
            g["lr"] = lr

        x, y = get_batch(train_data, args.batch_size, args.context_length, device)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=bool(args.amp)):
            logits = model(x)
            loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        train_loss = loss.item()
        if not math.isfinite(train_loss):
            diverged = True
            print(f"[train] step {step}: non-finite loss ({train_loss}); marking diverged and stopping.", flush=True)
            log({"step": step, "wall_clock_sec": round(time.time() - wall_start, 2), "train_loss": train_loss, "lr": lr, "diverged": True})
            break

        if step % args.log_every == 0:
            log({"step": step, "wall_clock_sec": round(time.time() - wall_start, 3), "train_loss": round(train_loss, 5), "lr": lr})
            print(f"[train] step {step} loss {train_loss:.4f} lr {lr:.2e} ({time.time()-wall_start:.0f}s)", flush=True)

        if val_data is not None and args.eval_every > 0 and (step + 1) % args.eval_every == 0:
            val_loss = evaluate(model, val_data, args, device)
            best_val = min(best_val, val_loss)
            final_val = val_loss
            log({"step": step, "wall_clock_sec": round(time.time() - wall_start, 3), "train_loss": round(train_loss, 5), "val_loss": round(val_loss, 5), "lr": lr})
            print(f"[train] step {step} VAL {val_loss:.4f} (best {best_val:.4f})", flush=True)

        if args.max_runtime_sec and (time.time() - wall_start) > args.max_runtime_sec:
            print(f"[train] hit max runtime {args.max_runtime_sec}s at step {step}", flush=True)
            break

    total_time = time.time() - wall_start

    if val_data is not None and not diverged:
        final_val = evaluate(model, val_data, args, device)
        best_val = min(best_val, final_val)

    if args.ckpt:
        os.makedirs(os.path.dirname(args.ckpt), exist_ok=True)
        save_checkpoint(model, optimizer, args.total_steps, args.ckpt)
        print(f"[train] saved checkpoint -> {args.ckpt}", flush=True)

    summary = {
        "config": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "total_steps": args.total_steps,
            "lr": args.lr,
            "min_lr": args.min_lr,
            "warmup_steps": args.warmup_steps,
            "weight_decay": args.weight_decay,
            "norm": args.norm,
            "norm_position": args.norm_position,
            "use_rope": bool(args.use_rope),
            "ffn": args.ffn,
            "vocab_size": args.vocab_size,
        },
        "parameters": n_params,
        "total_tokens": args.batch_size * args.context_length * (args.total_steps),
        "total_train_time_sec": round(total_time, 2),
        "final_val_loss": None if final_val is None else round(final_val, 5),
        "best_val_loss": None if best_val == math.inf else round(best_val, 5),
        "diverged": diverged,
    }
    print("[train] SUMMARY " + json.dumps(summary), flush=True)
    if args.summary_file:
        os.makedirs(os.path.dirname(args.summary_file), exist_ok=True)
        with open(args.summary_file, "w") as f:
            json.dump(summary, f, indent=2)
    if log_f:
        log_f.close()


if __name__ == "__main__":
    main()
