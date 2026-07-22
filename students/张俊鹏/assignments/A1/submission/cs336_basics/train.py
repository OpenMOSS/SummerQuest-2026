import argparse
import math
import os
import time

import numpy as np
import torch
import torch.nn as nn

from cs336_basics.transformer_lm import TransformerLM
from cs336_basics.get_batch import get_batch
from cs336_basics.learning_rate_schedule import learning_rate_schedule
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint
from cs336_basics.experiment import ExperimentTracker

# fmt: off
def parse_args():
    p = argparse.ArgumentParser(description="Train a Transformer language model.")
    # Data
    p.add_argument("--train_data", type=str, required=True, help="Path to training .npy memmap (int32)")
    p.add_argument("--val_data", type=str, required=True, help="Path to validation .npy memmap (int32)")
    # Model
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--theta", type=float, default=10000.0)
    # Optimizer / LR
    p.add_argument("--max_lr", type=float, default=1e-3)
    p.add_argument("--min_lr", type=float, default=1e-4)
    p.add_argument("--warmup_iters", type=int, default=100)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--beta1", type=float, default=0.9)
    p.add_argument("--beta2", type=float, default=0.95)
    p.add_argument("--eps", type=float, default=1e-8)
    # Training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--total_steps", type=int, default=20000)
    p.add_argument("--eval_every", type=int, default=500)
    p.add_argument("--log_every", type=int, default=10)
    # Checkpointing
    p.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    p.add_argument("--checkpoint_every", type=int, default=2000)
    p.add_argument("--resume", type=str, default=None)
    # Hardware
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--compile", action="store_true")
    # Experiment
    p.add_argument("--exp_name", type=str, default=None, help="Experiment name for logging")
    p.add_argument("--exp_dir", type=str, default="experiments", help="Root directory for experiment logs")
    p.add_argument("--wandb", action="store_true", help="Log to Weights & Biases")
    return p.parse_args()
# fmt: on


# def load_data(path):
#     return np.memmap(path, dtype=np.int32, mode="r")
def load_data(path: str) -> np.ndarray:
    """以只读内存映射方式加载标准 .npy 文件。"""
    return np.load(path, mmap_mode="r", allow_pickle=False)


@torch.no_grad()
def evaluate(model, val_tokens, batch_size, context_length, device, eval_batches=50):
    model.eval()
    total_loss = 0.0
    for _ in range(eval_batches):
        x, y = get_batch(val_tokens, batch_size, context_length, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        total_loss += loss.item()
    model.train()
    return total_loss / eval_batches


def main():
    args = parse_args()
    os.makedirs(args.checkpoint_dir, exist_ok=True)

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        print("CUDA not available, falling back to CPU")
        device = "cpu"

    # Experiment tracker (7.1)
    exp_name = args.exp_name or f"run_{int(time.time())}"
    tracker = ExperimentTracker(
        name=exp_name,
        config=vars(args),
        log_dir=args.exp_dir,
        use_wandb=args.wandb,
    )
    print(f"Experiment: {exp_name}  →  {tracker._log_dir}")

    # Load data via memmap
    print(f"Loading training data from {args.train_data}...")
    train_tokens = load_data(args.train_data)
    print(f"Loading validation data from {args.val_data}...")
    val_tokens = load_data(args.val_data)
    print(f"Train tokens: {len(train_tokens):,}  Val tokens: {len(val_tokens):,}")

    # Build model
    model = TransformerLM(
        vocab_size=args.vocab_size, context_length=args.context_length,
        num_layers=args.num_layers, d_model=args.d_model,
        num_heads=args.num_heads, d_ff=args.d_ff, theta=args.theta,
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr,
        betas=(args.beta1, args.beta2), eps=args.eps, weight_decay=args.weight_decay,
    )

    if args.compile:
        backend = "aot_eager" if device == "mps" else "inductor"
        model = torch.compile(model, backend=backend)
        print(f"Model compiled (backend={backend}).")

    start_step = 0
    if args.resume:
        print(f"Resuming from {args.resume}...")
        start_step = load_checkpoint(args.resume, model, optimizer) + 1

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total params: {n_params:,}")
    print(f"Training: step {start_step} → {args.total_steps}")
    print(f"{'='*60}")

    model.train()
    total_tokens = 0

    for step in range(start_step, args.total_steps):
        lr = learning_rate_schedule(step, args.max_lr, args.min_lr,
                                    args.warmup_iters, args.total_steps)
        for pg in optimizer.param_groups:
            pg["lr"] = lr

        x, y = get_batch(train_tokens, args.batch_size, args.context_length, device)
        logits = model(x)
        loss = nn.functional.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_tokens += args.batch_size * args.context_length

        if step % args.log_every == 0:
            print(f"step {step:6d}/{args.total_steps} | loss {loss.item():.4f} | lr {lr:.2e}")

        # Evaluation + logging (7.1)
        if step % args.eval_every == 0 and step > 0:
            val_loss = evaluate(model, val_tokens, args.batch_size, args.context_length, device)
            tracker.log(step=step, train_loss=loss.item(), val_loss=val_loss, lr=lr,
                        tokens_processed=total_tokens)
            print(f"  >> eval step {step:6d} | val_loss {val_loss:.4f} | perp {math.exp(val_loss):.2f}")

        if step % args.checkpoint_every == 0 and step > 0:
            ckpt_path = os.path.join(args.checkpoint_dir, f"step_{step:06d}.pt")
            save_checkpoint(model, optimizer, step, ckpt_path)
            print(f"  >> checkpoint → {ckpt_path}")

    # Final
    final_ckpt = os.path.join(args.checkpoint_dir, "final.pt")
    save_checkpoint(model, optimizer, args.total_steps - 1, final_ckpt)
    print(f"Final checkpoint → {final_ckpt}")
    print(tracker.summary())
    print("Training complete.")


if __name__ == "__main__":
    main()