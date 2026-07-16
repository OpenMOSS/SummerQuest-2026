from __future__ import annotations

import argparse
import json
import math
import time
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch

from cs336_basics.transformer import TransformerLM


def cross_entropy(inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Average cross-entropy over all batch-like dimensions."""

    max_value = torch.max(inputs, dim=-1, keepdim=True).values
    shifted = inputs - max_value
    log_sum_exp = torch.log(torch.sum(torch.exp(shifted), dim=-1)) + max_value.squeeze(-1)
    target_logits = torch.gather(inputs, dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)
    return torch.mean(log_sum_exp - target_logits)


class AdamW(torch.optim.Optimizer):
    """AdamW optimizer following the assignment pseudocode."""

    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.0,
    ):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0 <= betas[0] < 1:
            raise ValueError(f"Invalid beta1 value: {betas[0]}")
        if not 0 <= betas[1] < 1:
            raise ValueError(f"Invalid beta2 value: {betas[1]}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = {"lr": lr, "betas": betas, "eps": eps, "weight_decay": weight_decay}
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]

            for param in group["params"]:
                if param.grad is None:
                    continue

                grad = param.grad
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients.")

                state = self.state[param]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(param)
                    state["exp_avg_sq"] = torch.zeros_like(param)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                state["step"] += 1
                step = state["step"]

                with torch.no_grad():
                    lr_t = lr * math.sqrt(1 - beta2**step) / (1 - beta1**step)

                    if weight_decay != 0:
                        param.data = param.data - lr * weight_decay * param.data

                    exp_avg = beta1 * exp_avg + (1 - beta1) * grad
                    exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad**2
                    state["exp_avg"] = exp_avg
                    state["exp_avg_sq"] = exp_avg_sq

                    param.data = param.data - lr_t * exp_avg / (torch.sqrt(exp_avg_sq) + eps)

        return loss


def get_lr_cosine_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int,
) -> float:
    if it < warmup_iters:
        return it / warmup_iters * max_learning_rate
    if it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1 + math.cos(math.pi * progress))
        return min_learning_rate + cosine * (max_learning_rate - min_learning_rate)
    return min_learning_rate


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6) -> None:
    grads = [param.grad for param in parameters if param.grad is not None]
    if not grads:
        return

    total_norm = torch.sqrt(sum(torch.sum(grad.detach() ** 2) for grad in grads))
    clip_coef = max_l2_norm / (total_norm + eps)
    if clip_coef < 1:
        for grad in grads:
            grad.data = grad.data * clip_coef.to(device=grad.device, dtype=grad.dtype)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:

    if dataset.ndim != 1:
        raise ValueError("dataset must be a 1D array of token IDs.")
    if len(dataset) <= context_length:
        raise ValueError("dataset must be longer than context_length.")

    inputs = np.empty((batch_size, context_length), dtype=dataset.dtype)
    targets = np.empty((batch_size, context_length), dtype=dataset.dtype)

    for row in range(batch_size):
        start = np.random.randint(0, len(dataset) - context_length)
        inputs[row] = dataset[start : start + context_length]
        targets[row] = dataset[start + 1 : start + context_length + 1]

    x = torch.tensor(inputs, dtype=torch.long, device=device)
    y = torch.tensor(targets, dtype=torch.long, device=device)
    return x, y


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out,
) -> None:

    checkpoint = {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(
    src,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
) -> int:

    checkpoint = torch.load(src, map_location="cpu")
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return int(checkpoint["iteration"])


def load_token_array(path, mmap: bool = True) -> np.ndarray:
    """Load a `.npy` token array, using memory mapping by default."""

    return np.load(path, mmap_mode="r" if mmap else None)


@torch.no_grad()
def estimate_loss(
    model: torch.nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str | torch.device,
    eval_iters: int,
) -> float:
    """Estimate validation loss by averaging several sampled batches."""

    model.eval()
    losses = []
    for _ in range(eval_iters):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        losses.append(cross_entropy(logits, y).item())
    model.train()
    return float(np.mean(losses))


def train(args: argparse.Namespace) -> None:
    """Run a configurable Transformer LM training loop."""

    device = torch.device(args.device)
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        if device.type == "cuda":
            torch.cuda.manual_seed_all(args.seed)

    log_path = Path(args.log_jsonl) if args.log_jsonl is not None else None
    summary_path = Path(args.summary_json) if args.summary_json is not None else None
    checkpoint_path = Path(args.checkpoint_path) if args.checkpoint_path is not None else None
    for path in (log_path, summary_path, checkpoint_path):
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)

    train_data = load_token_array(args.train_data, mmap=True)
    val_data = load_token_array(args.val_data, mmap=True) if args.val_data is not None else None

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        post_norm=args.post_norm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    start_iter = 0
    if args.resume_from is not None:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"resumed checkpoint from iteration {start_iter}")

    log_file = log_path.open("a", encoding="utf-8") if log_path is not None else None
    start_time = time.time()
    latest_train_loss = None
    latest_val_loss = None
    total_tokens = 0
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    model.train()
    try:
        for iteration in range(start_iter, args.max_iters):
            completed = iteration + 1
            lr = get_lr_cosine_schedule(
                iteration,
                args.max_lr,
                args.min_lr,
                args.warmup_iters,
                args.max_iters,
            )
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            loss_for_log = None
            for _ in range(args.grad_accum_steps):
                x, y = get_batch(train_data, args.batch_size, args.context_length, device)
                logits = model(x)
                loss = cross_entropy(logits, y) / args.grad_accum_steps
                loss.backward()
                loss_for_log = loss.detach() * args.grad_accum_steps
                total_tokens += args.batch_size * args.context_length

            if args.grad_clip is not None:
                gradient_clipping(model.parameters(), args.grad_clip)
            optimizer.step()

            latest_train_loss = float(loss_for_log.item()) if loss_for_log is not None else None
            should_log = completed % args.log_every == 0 or completed == 1
            should_eval = val_data is not None and (completed % args.eval_every == 0 or completed == args.max_iters)
            if should_eval:
                latest_val_loss = estimate_loss(
                    model,
                    val_data,
                    args.eval_batch_size or args.batch_size,
                    args.context_length,
                    device,
                    args.eval_iters,
                )

            if should_log or should_eval:
                elapsed = time.time() - start_time
                record = {
                    "run_name": args.run_name,
                    "step": completed,
                    "wall_clock_sec": round(elapsed, 3),
                    "processed_tokens": total_tokens,
                    "train_loss": latest_train_loss,
                    "val_loss": latest_val_loss if should_eval else None,
                    "lr": lr,
                    "batch_size": args.batch_size,
                    "grad_accum_steps": args.grad_accum_steps,
                    "context_length": args.context_length,
                }
                if device.type == "cuda":
                    record["cuda_max_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
                print(json.dumps(record, ensure_ascii=False))
                if log_file is not None:
                    log_file.write(json.dumps(record, ensure_ascii=False) + "\n")
                    log_file.flush()

            if checkpoint_path is not None and completed % args.save_every == 0:
                save_checkpoint(model, optimizer, completed, checkpoint_path)
                print(f"saved checkpoint to {checkpoint_path}")

        if checkpoint_path is not None:
            save_checkpoint(model, optimizer, args.max_iters, checkpoint_path)
            print(f"saved final checkpoint to {checkpoint_path}")
    finally:
        if log_file is not None:
            log_file.close()

    if summary_path is not None:
        summary = {
            "run_name": args.run_name,
            "final_step": args.max_iters,
            "final_train_loss": latest_train_loss,
            "final_val_loss": latest_val_loss,
            "total_training_time_sec": round(time.time() - start_time, 3),
            "processed_tokens": total_tokens,
            "config": training_config_dict(args),
        }
        if device.type == "cuda":
            summary["cuda_max_memory_mb"] = round(torch.cuda.max_memory_allocated(device) / 1024**2, 2)
        summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def training_config_dict(args: argparse.Namespace) -> dict[str, object]:
    keys = [
        "train_data",
        "val_data",
        "device",
        "vocab_size",
        "context_length",
        "d_model",
        "num_layers",
        "num_heads",
        "d_ff",
        "rope_theta",
        "batch_size",
        "grad_accum_steps",
        "max_iters",
        "max_lr",
        "min_lr",
        "warmup_iters",
        "beta1",
        "beta2",
        "eps",
        "weight_decay",
        "grad_clip",
        "eval_every",
        "eval_iters",
        "no_rmsnorm",
        "post_norm",
        "no_rope",
        "ffn_type",
        "seed",
    ]
    return {key: getattr(args, key) for key in keys}


def load_config_file(path: str | None) -> dict[str, object]:
    if path is None:
        return {}
    config_path = Path(path)
    with config_path.open(encoding="utf-8") as f:
        return json.load(f)


def apply_config_defaults(parser: argparse.ArgumentParser, config: dict[str, object]) -> None:
    if not config:
        return
    valid_dests = {action.dest for action in parser._actions}
    defaults = {key.replace("-", "_"): value for key, value in config.items() if key.replace("-", "_") in valid_dests}
    parser.set_defaults(**defaults)


def parse_args() -> argparse.Namespace:
    parser = build_arg_parser(add_config_only=True)
    config_args, remaining = parser.parse_known_args()
    config = load_config_file(config_args.config)

    parser = build_arg_parser()
    apply_config_defaults(parser, config)
    args = parser.parse_args(remaining)
    missing = [name for name in ("train_data", "vocab_size") if getattr(args, name) is None]
    if missing:
        parser.error("missing required arguments: " + ", ".join(f"--{name.replace('_', '-')}" for name in missing))
    return args


def build_arg_parser(add_config_only: bool = False) -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a CS336 basics Transformer LM.")
    parser.add_argument("--config", default=None, help="Optional JSON config file.")
    if add_config_only:
        return parser

    parser.add_argument("--train-data", default=None, help="Path to tokenized training `.npy` file.")
    parser.add_argument("--val-data", default=None, help="Path to tokenized validation `.npy` file.")
    parser.add_argument("--checkpoint-path", default=None, help="Where to write checkpoints.")
    parser.add_argument("--resume-from", default=None, help="Checkpoint path to resume from.")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--run-name", default="train")
    parser.add_argument("--log-jsonl", default=None, help="Where to append JSONL training records.")
    parser.add_argument("--summary-json", default=None, help="Where to write final run summary.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument("--vocab-size", type=int, default=None)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--post-norm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")

    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batch-size", type=int, default=None)
    parser.add_argument("--grad-accum-steps", type=int, default=1)
    parser.add_argument("--max-iters", type=int, default=1000)
    parser.add_argument("--max-lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)

    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-iters", type=int, default=20)
    parser.add_argument("--save-every", type=int, default=1000)
    return parser


def main() -> None:
    train(parse_args())


if __name__ == "__main__":
    main()
