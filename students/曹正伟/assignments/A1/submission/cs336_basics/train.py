"""Configurable Transformer language-model training entry point."""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradients, cross_entropy
from cs336_basics.optimizer import AdamW, cosine_learning_rate


def load_token_data(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    """Memory-map either a ``.npy`` array or a raw one-dimensional token file."""
    path = Path(path)
    if path.suffix == ".npy":
        data = np.load(path, mmap_mode="r")
    else:
        data = np.memmap(path, mode="r", dtype=np.dtype(dtype))
    if data.ndim != 1:
        raise ValueError(f"token data must be one-dimensional, got {data.shape}")
    return data


@torch.inference_mode()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: torch.device,
    batches: int,
) -> float:
    """Estimate mean validation loss over independently sampled batches."""
    was_training = model.training
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(float(cross_entropy(logits, targets).item()))
    model.train(was_training)
    return sum(losses) / len(losses)


def save_training_checkpoint(
    path: Path,
    model: TransformerLM,
    optimizer: AdamW,
    iteration: int,
    model_config: dict[str, Any],
    best_validation_loss: float,
) -> None:
    """Write a resumable training checkpoint atomically within one directory."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "iteration": iteration,
            "model_config": model_config,
            "best_validation_loss": best_validation_loss,
        },
        temporary_path,
    )
    temporary_path.replace(path)


def append_jsonl(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as file:
        file.write(json.dumps(record, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-data", type=Path, required=True)
    parser.add_argument("--validation-data", type=Path, required=True)
    parser.add_argument("--data-dtype", choices=("uint16", "uint32", "int32", "int64"), default="uint16")
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10_000.0)
    parser.add_argument("--normalization", choices=("rmsnorm", "none"), default="rmsnorm")
    parser.add_argument("--position-encoding", choices=("rope", "none"), default="rope")
    parser.add_argument("--ffn-type", choices=("swiglu", "silu"), default="swiglu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--steps", type=int, default=10_000)
    parser.add_argument("--max-learning-rate", type=float, default=3e-4)
    parser.add_argument("--min-learning-rate", type=float, default=3e-5)
    parser.add_argument("--warmup-steps", type=int, default=200)
    parser.add_argument("--cosine-cycle-steps", type=int)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.95)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--eval-interval", type=int, default=100)
    parser.add_argument("--eval-batches", type=int, default=10)
    parser.add_argument("--checkpoint-interval", type=int, default=500)
    parser.add_argument("--checkpoint", type=Path, default=Path("checkpoints/latest.pt"))
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--log-path", type=Path, default=Path("training_log.jsonl"))
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if args.steps <= 0 or args.eval_interval <= 0 or args.checkpoint_interval <= 0:
        parser.error("steps and intervals must be positive")
    cosine_cycle_steps = args.cosine_cycle_steps or args.steps
    device = torch.device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    train_data = load_token_data(args.train_data, args.data_dtype)
    validation_data = load_token_data(args.validation_data, args.data_dtype)
    largest_token = max(int(train_data.max()), int(validation_data.max()))
    if largest_token >= args.vocab_size:
        raise ValueError(f"dataset contains token ID {largest_token}, but vocab_size is {args.vocab_size}")

    model_config = {
        "vocab_size": args.vocab_size,
        "context_length": args.context_length,
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "rope_theta": args.rope_theta,
        "normalization": args.normalization,
        "position_encoding": args.position_encoding,
        "ffn_type": args.ffn_type,
    }
    model = TransformerLM(**model_config, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=args.max_learning_rate,
        betas=(args.beta1, args.beta2),
        weight_decay=args.weight_decay,
    )
    start_iteration = 0
    best_validation_loss = math.inf
    if args.resume is not None:
        checkpoint = torch.load(args.resume, map_location=device, weights_only=False)
        if checkpoint.get("model_config") != model_config:
            raise ValueError("resume checkpoint model configuration does not match CLI arguments")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_iteration = int(checkpoint["iteration"])
        best_validation_loss = float(checkpoint.get("best_validation_loss", math.inf))

    model.train()
    started = time.perf_counter()
    for iteration in range(start_iteration, args.steps):
        learning_rate = cosine_learning_rate(
            iteration,
            args.max_learning_rate,
            args.min_learning_rate,
            args.warmup_steps,
            cosine_cycle_steps,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        inputs, targets = get_batch(train_data, args.batch_size, args.context_length, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        loss.backward()
        clip_gradients(model.parameters(), args.grad_clip)
        optimizer.step()

        completed_steps = iteration + 1
        if completed_steps == 1 or completed_steps % args.eval_interval == 0:
            validation_loss = evaluate(
                model,
                validation_data,
                args.batch_size,
                args.context_length,
                device,
                args.eval_batches,
            )
            elapsed = time.perf_counter() - started
            record = {
                "iteration": completed_steps,
                "elapsed_seconds": elapsed,
                "train_loss": float(loss.item()),
                "validation_loss": validation_loss,
                "validation_perplexity": math.exp(min(validation_loss, 50.0)),
                "learning_rate": learning_rate,
                "tokens_seen": completed_steps * args.batch_size * args.context_length,
            }
            append_jsonl(args.log_path, record)
            print(json.dumps(record, ensure_ascii=False))
            if validation_loss < best_validation_loss:
                best_validation_loss = validation_loss
                save_training_checkpoint(
                    args.checkpoint.with_name("best.pt"),
                    model,
                    optimizer,
                    completed_steps,
                    model_config,
                    best_validation_loss,
                )

        if completed_steps % args.checkpoint_interval == 0 or completed_steps == args.steps:
            save_training_checkpoint(
                args.checkpoint,
                model,
                optimizer,
                completed_steps,
                model_config,
                best_validation_loss,
            )


if __name__ == "__main__":
    main()
