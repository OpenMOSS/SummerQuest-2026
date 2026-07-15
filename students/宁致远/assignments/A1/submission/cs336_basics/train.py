"""Training driver: reads uint16 token arrays, trains TransformerLM, logs JSONL."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch

from .model import TransformerLM, cross_entropy
from .optim import AdamW, clip_grad_l2, cosine_lr, get_batch, load_checkpoint, save_checkpoint


def _load_tokens(path: str, dtype=np.uint16) -> np.ndarray:
    return np.memmap(path, dtype=dtype, mode="r")


@torch.no_grad()
def evaluate(model: TransformerLM, data: np.ndarray, batch_size: int, context_length: int, device: str, iters: int) -> float:
    model.eval()
    total = 0.0
    for _ in range(iters):
        x, y = get_batch(data, batch_size, context_length, device)
        loss = cross_entropy(model(x), y)
        total += float(loss.item())
    model.train()
    return total / iters


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--vocab-size", type=int, required=True)
    ap.add_argument("--context-length", type=int, default=256)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--d-ff", type=int, default=1344)
    ap.add_argument("--num-layers", type=int, default=4)
    ap.add_argument("--num-heads", type=int, default=16)
    ap.add_argument("--rope-theta", type=float, default=10000.0)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--total-steps", type=int, default=10_000)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--min-lr", type=float, default=3e-5)
    ap.add_argument("--warmup", type=int, default=200)
    ap.add_argument("--weight-decay", type=float, default=0.1)
    ap.add_argument("--beta1", type=float, default=0.9)
    ap.add_argument("--beta2", type=float, default=0.95)
    ap.add_argument("--grad-clip", type=float, default=1.0)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-interval", type=int, default=500)
    ap.add_argument("--val-iters", type=int, default=20)
    ap.add_argument("--ckpt-interval", type=int, default=1000)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--resume", type=str, default=None)
    ap.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--dtype", type=str, default="float32", choices=["float32", "bfloat16"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--variant", type=str, default="baseline",
                    help="baseline | no_rmsnorm | post_norm | nope | silu_ffn")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_data = _load_tokens(args.train)
    val_data = _load_tokens(args.val)

    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16}[args.dtype]
    if args.variant == "baseline":
        model = TransformerLM(
            args.vocab_size, args.context_length, args.d_model, args.num_layers,
            args.num_heads, args.d_ff, args.rope_theta, device=args.device, dtype=dtype,
        )
    else:
        from .variants import build_variant
        model = build_variant(args.variant, args, dtype)

    if args.compile:
        model = torch.compile(model)

    optim = AdamW(model.parameters(), lr=args.lr, betas=(args.beta1, args.beta2),
                  weight_decay=args.weight_decay)

    start_step = 0
    if args.resume:
        start_step = load_checkpoint(args.resume, model, optim)

    log_path = out / "train.jsonl"
    log_f = log_path.open("a")
    t0 = time.time()
    processed = 0

    def log(rec):
        log_f.write(json.dumps(rec) + "\n")
        log_f.flush()

    n_params = sum(p.numel() for p in model.parameters())
    (out / "config.json").write_text(json.dumps({**vars(args), "n_params": n_params}, indent=2))

    print(f"params={n_params/1e6:.2f}M device={args.device}")
    for step in range(start_step, args.total_steps):
        lr = cosine_lr(step, args.lr, args.min_lr, args.warmup, args.total_steps)
        for g in optim.param_groups:
            g["lr"] = lr
        x, y = get_batch(train_data, args.batch_size, args.context_length, args.device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        optim.zero_grad(set_to_none=True)
        loss.backward()
        clip_grad_l2(model.parameters(), args.grad_clip)
        optim.step()
        processed += args.batch_size * args.context_length

        if step % args.log_interval == 0:
            log({"step": step, "wall": round(time.time() - t0, 2), "train_loss": float(loss.item()),
                 "lr": lr, "processed_tokens": processed})
            print(f"step={step} loss={float(loss.item()):.4f} lr={lr:.2e} t={time.time()-t0:.0f}s")

        if step > 0 and step % args.val_interval == 0:
            vloss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.val_iters)
            log({"step": step, "wall": round(time.time() - t0, 2), "val_loss": vloss,
                 "processed_tokens": processed})
            print(f"  val_loss={vloss:.4f}")

        if step > 0 and step % args.ckpt_interval == 0:
            save_checkpoint(model, optim, step, out / f"ckpt-{step}.pt")

    vloss = evaluate(model, val_data, args.batch_size, args.context_length, args.device, args.val_iters)
    total_time = time.time() - t0
    log({"step": args.total_steps, "wall": round(total_time, 2), "val_loss": vloss,
         "processed_tokens": processed, "final": True})
    save_checkpoint(model, optim, args.total_steps, out / "ckpt-final.pt")
    (out / "summary.json").write_text(json.dumps({
        "final_val_loss": vloss, "total_time_sec": total_time,
        "processed_tokens": processed, "n_params": n_params,
        **{k: getattr(args, k) for k in ["d_model", "num_layers", "num_heads",
             "context_length", "batch_size", "total_steps", "vocab_size", "variant"]},
    }, indent=2))
    log_f.close()
    print(f"done. final val_loss={vloss:.4f}")


if __name__ == "__main__":
    main()
