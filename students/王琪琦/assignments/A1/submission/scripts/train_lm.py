from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradients, cross_entropy
from cs336_basics.optimizer import AdamW, cosine_learning_rate


@torch.no_grad()
def validate(
    model: TransformerLM,
    dataset: np.memmap,
    batches: int,
    batch_size: int,
    context_length: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        losses.append(cross_entropy(model(inputs), targets).item())
    model.train()
    return sum(losses) / len(losses)


def read_config(path: str) -> tuple[dict[str, Any], dict[str, Any]]:
    config = json.loads(Path(path).read_text(encoding="utf-8"))
    return config["model"], config["training"]


def previous_wall_clock(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    last_record = None
    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                last_record = json.loads(line)
    return float(last_record["wall_clock_sec"]) if last_record else 0.0


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer LM.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--valid-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--data-dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume")
    args = parser.parse_args()

    model_config, training = read_config(args.config)
    device = torch.device(args.device)
    torch.manual_seed(training["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed_all(training["seed"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.jsonl"

    train_data = np.memmap(args.train_data, mode="r", dtype=args.data_dtype)
    valid_data = np.memmap(args.valid_data, mode="r", dtype=args.data_dtype)
    model = TransformerLM(**model_config, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=training["max_learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    start_step = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    prior_wall_clock = previous_wall_clock(log_path) if args.resume else 0.0
    start_time = time.perf_counter() - prior_wall_clock
    final_val_loss: float | None = None

    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(start_step + 1, training["steps"] + 1):
            learning_rate = cosine_learning_rate(
                step - 1,
                training["max_learning_rate"],
                training["min_learning_rate"],
                training["warmup_iters"],
                training["cosine_cycle_iters"],
            )
            for group in optimizer.param_groups:
                group["lr"] = learning_rate

            inputs, targets = get_batch(
                train_data,
                training["batch_size"],
                model_config["context_length"],
                device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(inputs), targets)
            loss.backward()
            clip_gradients(model.parameters(), training["max_grad_norm"])
            optimizer.step()

            should_validate = step % training["validation_interval"] == 0 or step == training["steps"]
            if should_validate:
                final_val_loss = validate(
                    model,
                    valid_data,
                    training["validation_batches"],
                    training["batch_size"],
                    model_config["context_length"],
                    device,
                )
            if step % training["log_interval"] == 0 or should_validate or step == 1:
                record = {
                    "step": step,
                    "wall_clock_sec": time.perf_counter() - start_time,
                    "train_loss": loss.item(),
                    "lr": learning_rate,
                }
                if final_val_loss is not None and should_validate:
                    record["val_loss"] = final_val_loss
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record))
            if step % training["checkpoint_interval"] == 0 or step == training["steps"]:
                save_checkpoint(model, optimizer, step, output_dir / "checkpoint_latest.pt")

    total_seconds = time.perf_counter() - start_time
    summary = {
        **model_config,
        "batch_size": training["batch_size"],
        "total_steps": training["steps"],
        "processed_tokens": training["steps"] * training["batch_size"] * model_config["context_length"],
        "final_val_loss": final_val_loss,
        "total_training_time_sec": total_seconds,
        "device": str(device),
    }
    (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
