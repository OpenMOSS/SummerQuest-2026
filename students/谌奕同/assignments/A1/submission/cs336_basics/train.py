"""Train a Transformer language model."""

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.tokenizer import Tokenizer


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample a language modeling batch from a tokenized dataset."""
    n = len(dataset)
    starts_np = np.random.randint(0, n - context_length, size=(batch_size,))
    x = np.stack([dataset[s : s + context_length] for s in starts_np])
    y = np.stack([dataset[s + 1 : s + context_length + 1] for s in starts_np])
    return torch.from_numpy(x).long().to(device), torch.from_numpy(y).long().to(device)


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy loss over flattened batch x sequence positions."""
    logits = logits.reshape(-1, logits.size(-1))
    targets = targets.reshape(-1)
    log_sum_exp = torch.logsumexp(logits, dim=-1)
    target_logits = logits.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return (log_sum_exp - target_logits).mean()


def gradient_clipping(parameters, max_l2_norm: float, eps: float = 1e-6) -> float:
    """Clip the combined L2 norm of gradients in-place. Returns total norm."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return 0.0

    # Use foreach ops for speed when all grads share device/dtype.
    device_dtype_groups = {}
    for g in grads:
        key = (str(g.device), g.dtype)
        device_dtype_groups.setdefault(key, []).append(g)

    total_norm_sq = 0.0
    for subgroup in device_dtype_groups.values():
        norms = torch._foreach_norm(subgroup, 2)
        total_norm_sq += sum(n.item() ** 2 for n in norms)

    total_norm = math.sqrt(total_norm_sq)
    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for subgroup in device_dtype_groups.values():
            torch._foreach_mul_(subgroup, clip_coef)
    return total_norm


def get_lr(it: int, config: dict) -> float:
    """Cosine learning rate schedule with linear warmup."""
    max_lr = config["max_learning_rate"]
    min_lr = config["min_learning_rate"]
    warmup_iters = config["warmup_iters"]
    cosine_cycle_iters = config["cosine_cycle_iters"]

    if it < warmup_iters:
        return max_lr * (it / warmup_iters)
    if it >= cosine_cycle_iters:
        return min_lr
    progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    return min_lr + (max_lr - min_lr) * cosine


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    num_batches: int,
) -> float:
    """Compute average cross-entropy loss on validation samples."""
    model.eval()
    losses = []
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        losses.append(loss.item())
    model.train()
    return float(np.mean(losses))


def save_checkpoint(model, optimizer, iteration: int, config: dict, out: Path) -> None:
    state_dict = model.state_dict()
    # torch.compile wraps the model in OptimizedModule; strip the prefix so
    # checkpoints can be loaded into a plain TransformerLM.
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        state_dict = {k.removeprefix("_orig_mod."): v for k, v in state_dict.items()}
    checkpoint = {
        "model_state_dict": state_dict,
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
        "config": config,
    }
    torch.save(checkpoint, out)


def main():
    parser = argparse.ArgumentParser(description="Train a Transformer LM.")
    # Data
    parser.add_argument("--train_tokens", type=str, required=True)
    parser.add_argument("--val_tokens", type=str, required=True)
    parser.add_argument("--vocab_path", type=str, required=True)
    parser.add_argument("--merges_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./checkpoints")
    # Model
    parser.add_argument("--vocab_size", type=int, default=10_000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--rope_theta", type=float, default=10_000.0)
    parser.add_argument("--use_rmsnorm", action="store_true", default=True)
    parser.add_argument("--no_rmsnorm", dest="use_rmsnorm", action="store_false")
    parser.add_argument("--use_post_norm", action="store_true", default=False)
    parser.add_argument("--use_rope", action="store_true", default=True)
    parser.add_argument("--no_rope", dest="use_rope", action="store_false")
    parser.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"])
    parser.add_argument("--qk_norm", action="store_true", default=False, help="Apply RMSNorm to Q and K before RoPE.")
    parser.add_argument("--zero_init_output", action="store_true", default=False, help="Zero-initialize attention output and FFN output projections.")
    # Optimizer / training
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--max_iters", type=int, default=5_000)
    parser.add_argument("--learning_rate", type=float, default=6e-4)
    parser.add_argument("--min_learning_rate", type=float, default=6e-5)
    parser.add_argument("--warmup_iters", type=int, default=500)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--grad_accum_steps", type=int, default=1, help="Number of gradient accumulation steps per optimizer update.")
    # Logging / checkpointing
    parser.add_argument("--eval_interval", type=int, default=500)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--checkpoint_interval", type=int, default=2_500)
    parser.add_argument("--log_interval", type=int, default=100)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    torch.set_float32_matmul_precision("high")
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log_path = output_dir / "train.log"
    log_file = open(log_path, "w")

    def log(msg: str) -> None:
        print(msg)
        log_file.write(msg + "\n")
        log_file.flush()

    # Save config for reproducibility.
    with open(output_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    tokenizer = Tokenizer.from_files(
        args.vocab_path, args.merges_path, special_tokens=["<|endoftext|>"]
    )

    log(f"Loading datasets from {args.train_tokens} and {args.val_tokens}")
    train_data = np.load(args.train_tokens, mmap_mode="r")
    val_data = np.load(args.val_tokens, mmap_mode="r")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=args.use_rmsnorm,
        use_post_norm=args.use_post_norm,
        use_rope=args.use_rope,
        ffn_type=args.ffn_type,
        qk_norm=args.qk_norm,
        zero_init_output=args.zero_init_output,
    ).to(device)
    model = torch.compile(model)
    log("Model compiled with torch.compile")
    log(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    model.train()
    start_time = time.time()
    last_eval_time = start_time
    best_val_loss = float("inf")
    tokens_per_step = args.batch_size * args.context_length * args.grad_accum_steps

    for it in range(args.max_iters):
        lr = get_lr(it, {
            "max_learning_rate": args.learning_rate,
            "min_learning_rate": args.min_learning_rate,
            "warmup_iters": args.warmup_iters,
            "cosine_cycle_iters": args.max_iters,
        })
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        optimizer.zero_grad()
        accum_loss = 0.0
        for micro_step in range(args.grad_accum_steps):
            x, y = get_batch(train_data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy(logits, y)
            loss_scaled = loss / args.grad_accum_steps
            loss_scaled.backward()
            accum_loss += loss.item()

        avg_loss = accum_loss / args.grad_accum_steps
        grad_norm = gradient_clipping(model.parameters(), args.grad_clip)
        optimizer.step()

        if it % args.log_interval == 0 or it == args.max_iters - 1:
            elapsed = time.time() - start_time
            throughput = tokens_per_step * (it + 1) / elapsed
            log(
                f"iter {it:5d} | train loss {avg_loss:.4f} | "
                f"lr {lr:.2e} | grad norm {grad_norm:.2f} | "
                f"tokens/s {throughput:,.0f}"
            )

        if it % args.eval_interval == 0 or it == args.max_iters - 1:
            elapsed = time.time() - last_eval_time
            interval_steps = args.eval_interval if it > 0 else 1
            throughput = tokens_per_step * interval_steps / elapsed
            val_loss = evaluate(
                model, val_data, args.batch_size, args.context_length, device, args.eval_batches
            )
            log(
                f"iter {it:5d} | train loss {avg_loss:.4f} | val loss {val_loss:.4f} | "
                f"lr {lr:.2e} | grad norm {grad_norm:.2f} | "
                f"step tok/s {throughput:,.0f} | time {time.time() - start_time:.1f}s"
            )
            last_eval_time = time.time()
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(model, optimizer, it, vars(args), output_dir / "best.pt")

        if it > 0 and it % args.checkpoint_interval == 0:
            save_checkpoint(model, optimizer, it, vars(args), output_dir / f"checkpoint_{it:06d}.pt")

    save_checkpoint(model, optimizer, args.max_iters, vars(args), output_dir / "final.pt")
    log(f"Training complete. Best val loss: {best_val_loss:.4f}")
    log_file.close()


if __name__ == "__main__":
    main()
