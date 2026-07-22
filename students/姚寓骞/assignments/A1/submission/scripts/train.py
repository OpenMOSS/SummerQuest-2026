from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cosine_schedule, cross_entropy, get_batch, gradient_clipping
from cs336_basics.optimizer import AdamW


@torch.no_grad()
def evaluate(model, data, batch_size, context_length, device, batches: int) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        x, y = get_batch(data, batch_size, context_length, device)
        losses.append(cross_entropy(model(x).flatten(0, 1), y.flatten()).item())
    model.train()
    return float(np.mean(losses))


def previous_wall_clock(log_path: Path) -> float:
    """Recover accumulated wall-clock time when resuming an existing run."""
    if not log_path.is_file():
        return 0.0
    elapsed = 0.0
    with log_path.open() as file:
        for line in file:
            if not line.strip():
                continue
            record = json.loads(line)
            # Accept logs emitted before the A1 26.0.4 field-name update.
            elapsed = max(elapsed, float(record.get("wall_clock_sec", record.get("wall_time", 0.0))))
    return elapsed


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer LM")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text())
    torch.manual_seed(config.get("seed", 42))
    np.random.seed(config.get("seed", 42))
    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")

    train_data = np.memmap(config["train_data"], dtype=np.uint16, mode="r")
    valid_data = np.memmap(config["valid_data"], dtype=np.uint16, mode="r")
    model = TransformerLM(**config["model"]).to(device)
    optimizer = AdamW(model.parameters(), **config["optimizer"])
    start_step = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "metrics.jsonl"
    summary_path = output_dir / "summary.json"
    if start_step == 0 and log_path.exists():
        raise FileExistsError(
            f"refusing to mix a new run with an existing log: {log_path}; choose a new output_dir or pass --resume"
        )
    elapsed_before_resume = previous_wall_clock(log_path) if args.resume else 0.0
    started = time.perf_counter()
    final_val_loss: float | None = None

    for step in range(start_step, config["max_steps"]):
        lr = cosine_schedule(step, config["max_lr"], config["min_lr"], config["warmup_steps"], config["max_steps"])
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(train_data, config["batch_size"], config["model"]["context_length"], device)
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(x).flatten(0, 1), y.flatten())
        loss.backward()
        gradient_clipping(model.parameters(), config["max_grad_norm"])
        optimizer.step()

        iteration = step + 1
        should_evaluate = iteration % config["eval_every"] == 0 or iteration == config["max_steps"]
        should_log = iteration % config["log_every"] == 0 or iteration == 1 or should_evaluate
        if should_log:
            record = {
                "step": iteration,
                "train_loss": loss.item(),
                "lr": lr,
                "processed_tokens": iteration * config["batch_size"] * config["model"]["context_length"],
            }
            if should_evaluate:
                final_val_loss = evaluate(
                    model,
                    valid_data,
                    config["batch_size"],
                    config["model"]["context_length"],
                    device,
                    config["eval_batches"],
                )
                record["val_loss"] = final_val_loss
            record["wall_clock_sec"] = elapsed_before_resume + time.perf_counter() - started
            with log_path.open("a") as file:
                file.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if iteration % config["checkpoint_every"] == 0 or iteration == config["max_steps"]:
            save_checkpoint(model, optimizer, iteration, output_dir / f"checkpoint_{iteration:07d}.pt")

    total_wall_clock = elapsed_before_resume + time.perf_counter() - started
    summary = {
        "status": "completed",
        "final_val_loss": final_val_loss,
        "total_wall_clock_sec": total_wall_clock,
        "d_model": config["model"]["d_model"],
        "num_layers": config["model"]["num_layers"],
        "num_heads": config["model"]["num_heads"],
        "context_length": config["model"]["context_length"],
        "batch_size": config["batch_size"],
        "total_steps": config["max_steps"],
        "processed_tokens": config["max_steps"] * config["batch_size"] * config["model"]["context_length"],
        "max_lr": config["max_lr"],
        "min_lr": config["min_lr"],
        "warmup_steps": config["warmup_steps"],
        "seed": config.get("seed", 42),
        "use_rmsnorm": config["model"].get("use_rmsnorm", True),
        "norm_position": config["model"].get("norm_position", "pre"),
        "position_encoding": config["model"].get("position_encoding", "rope"),
        "ffn_type": config["model"].get("ffn_type", "swiglu"),
        "d_ff": config["model"]["d_ff"],
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
