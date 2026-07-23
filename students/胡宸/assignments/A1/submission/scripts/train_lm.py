#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy
from cs336_basics.optim import AdamW
from cs336_basics.serialization import load_checkpoint, save_checkpoint
from cs336_basics.training import clip_gradients, cosine_lr_schedule, get_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer LM on binary token ids.")
    parser.add_argument("--train-data", required=True, type=Path)
    parser.add_argument("--valid-data", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
    parser.add_argument("--valid-dtype", choices=["uint16", "uint32"])
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--no-rmsnorm", action="store_true", help="Ablation: replace all RMSNorms with identity.")
    parser.add_argument("--norm-position", choices=["pre", "post"], default="pre", help="Ablation: pre-norm or post-norm blocks.")
    parser.add_argument("--no-rope", action="store_true", help="Ablation: disable RoPE in attention.")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu", help="Ablation: feed-forward variant.")
    parser.add_argument("--silu-d-ff", type=int, help="Hidden width for --ffn-type silu. Defaults to --d-ff.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--min-lr", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=100)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--log-every", type=int, default=10)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=16)
    parser.add_argument("--checkpoint-every", type=int, default=0)
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--no-mmap", action="store_true", help="Load token ids eagerly instead of using np.memmap.")
    parser.add_argument("--compile", action="store_true", help="Use torch.compile for the model forward/backward path.")
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 matmul on CUDA for faster float32 training.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return parser.parse_args()


def load_tokens(path: Path, dtype_name: str, mmap: bool = True) -> np.ndarray:
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32
    data = np.memmap(path, dtype=dtype, mode="r") if mmap else np.fromfile(path, dtype=dtype)
    if data.ndim != 1 or len(data) == 0:
        raise ValueError(f"{path} does not contain any token ids")
    return data


def compute_loss(model: TransformerLM, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    logits = model(x)
    return cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))


def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int,
) -> float:
    model.eval()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(num_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            losses.append(float(compute_loss(model, x, y).item()))
    model.train()
    return float(sum(losses) / len(losses))


def write_event(log_file: Path | None, event: dict[str, object]) -> None:
    line = json.dumps(event, ensure_ascii=False, sort_keys=True)
    print(line, flush=True)
    if log_file is not None:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        with log_file.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def checkpoint_path(output: Path, step: int) -> Path:
    return output.with_name(f"{output.stem}_step{step}{output.suffix}")


def unwrap_compiled_model(model: torch.nn.Module) -> torch.nn.Module:
    return getattr(model, "_orig_mod", model)


def main() -> int:
    args = parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested, but torch.cuda.is_available() is False")
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cuda.matmul.allow_tf32 = True
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    train_dataset = load_tokens(args.train_data, args.dtype, mmap=not args.no_mmap)
    valid_dataset = None
    if args.valid_data is not None:
        valid_dataset = load_tokens(args.valid_data, args.valid_dtype or args.dtype, mmap=not args.no_mmap)

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        norm_position=args.norm_position,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        silu_d_ff=args.silu_d_ff,
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=args.lr)
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
    if args.compile:
        model = torch.compile(model)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text("", encoding="utf-8")

    write_event(
        args.log_file,
        {
            "event": "start",
            "time": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "device": args.device,
            "seed": args.seed,
            "train_tokens": int(len(train_dataset)),
            "valid_tokens": int(len(valid_dataset)) if valid_dataset is not None else None,
            "start_step": start_step,
            "steps": args.steps,
            "vocab_size": args.vocab_size,
            "context_length": args.context_length,
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "d_ff": args.d_ff,
            "batch_size": args.batch_size,
            "use_rmsnorm": not args.no_rmsnorm,
            "norm_position": args.norm_position,
            "use_rope": not args.no_rope,
            "ffn_type": args.ffn_type,
            "silu_d_ff": args.silu_d_ff,
            "mmap": not args.no_mmap,
            "compile": args.compile,
            "tf32": args.tf32,
        },
    )

    model.train()
    last_loss = None
    start_time = time.time()
    for step in range(start_step, args.steps):
        lr = cosine_lr_schedule(step, args.lr, args.min_lr, args.warmup_steps, args.steps)
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_dataset, args.batch_size, args.context_length, args.device)
        loss = compute_loss(model, x, y)
        optimizer.zero_grad()
        loss.backward()
        if args.grad_clip > 0:
            clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()
        last_loss = float(loss.item())

        current_step = step + 1
        should_log = args.log_every > 0 and (current_step == 1 or current_step % args.log_every == 0)
        should_eval = (
            valid_dataset is not None
            and args.eval_every > 0
            and (current_step == 1 or current_step % args.eval_every == 0 or current_step == args.steps)
        )
        if should_log or should_eval:
            event: dict[str, object] = {
                "event": "step",
                "step": current_step,
                "train_loss": last_loss,
                "lr": lr,
                "elapsed_sec": round(time.time() - start_time, 3),
            }
            if should_eval and valid_dataset is not None:
                event["valid_loss"] = evaluate(
                    model,
                    valid_dataset,
                    args.batch_size,
                    args.context_length,
                    args.device,
                    args.eval_batches,
                )
            write_event(args.log_file, event)

        if args.checkpoint_every > 0 and current_step % args.checkpoint_every == 0 and current_step != args.steps:
            save_checkpoint(unwrap_compiled_model(model), optimizer, current_step, checkpoint_path(args.output, current_step))

    save_checkpoint(unwrap_compiled_model(model), optimizer, args.steps, args.output)
    write_event(
        args.log_file,
        {
            "event": "done",
            "step": args.steps,
            "train_loss": last_loss,
            "output": str(args.output),
            "elapsed_sec": round(time.time() - start_time, 3),
        },
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
