from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path
from typing import cast

import numpy as np
import torch
from torch import nn

from cs336_basics.training import (
    AdamW,
    clip_gradients,
    cosine_learning_rate,
    cross_entropy,
    estimate_loss,
    load_checkpoint,
    save_checkpoint,
)
from cs336_basics.training import get_batch
from cs336_basics.transformer import TransformerLM
from config_utils import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Assignment 1 Transformer LM from a JSON config.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path)
    return parser.parse_args()


def choose_device(requested: str) -> torch.device:
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")

    seed = int(config.get("seed", 42))
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    device = choose_device(config.get("device", "auto"))
    if device.type == "cuda":
        torch.set_float32_matmul_precision("high")
    data_dtype = np.dtype(config.get("data_dtype", "uint16"))
    train_data = np.memmap(config["train_path"], mode="r", dtype=data_dtype)
    val_data = np.memmap(config["val_path"], mode="r", dtype=data_dtype)

    model_config = config["model"]
    raw_model = TransformerLM(**model_config, device=device).to(device)
    optimizer_config = config["optimizer"]
    optimizer = AdamW(raw_model.parameters(), **optimizer_config)
    start_step = load_checkpoint(args.resume, raw_model, optimizer) if args.resume else 0
    model = cast(nn.Module, torch.compile(raw_model)) if config.get("compile", False) else raw_model

    precision = config.get("precision", "fp32")
    if precision == "bf16" and device.type == "cuda":
        amp_dtype: torch.dtype | None = torch.bfloat16
    elif precision == "fp16" and device.type == "cuda":
        amp_dtype = torch.float16
    elif precision == "fp32":
        amp_dtype = None
    else:
        raise ValueError(f"unsupported precision {precision!r} for device {device}")

    training = config["training"]
    max_steps = int(training["max_steps"])
    batch_size = int(training["batch_size"])
    context_length = int(model_config["context_length"])
    max_lr = float(optimizer_config["lr"])
    min_lr = float(training.get("min_lr", max_lr * 0.1))
    warmup_steps = int(training.get("warmup_steps", 0))
    eval_interval = int(training.get("eval_interval", 100))
    eval_batches = int(training.get("eval_batches", 10))
    eval_batch_size = int(training.get("eval_batch_size", batch_size))
    checkpoint_interval = int(training.get("checkpoint_interval", eval_interval))
    max_grad_norm = float(training.get("max_grad_norm", 1.0))
    log_path = args.output_dir / "metrics.jsonl"
    start_time = time.perf_counter()
    last_val_loss: float | None = None
    completed_steps = start_step
    run_status = "completed"
    divergence_reason: str | None = None

    model.train()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(start_step, max_steps):
            step_start = time.perf_counter()
            lr = cosine_learning_rate(step, max_lr, min_lr, warmup_steps, max_steps)
            for group in optimizer.param_groups:
                group["lr"] = lr
            inputs, targets = get_batch(train_data, batch_size, context_length, device)
            optimizer.zero_grad(set_to_none=True)
            amp_context = (
                torch.autocast(device_type=device.type, dtype=amp_dtype)
                if amp_dtype is not None
                else torch.autocast(device_type=device.type, enabled=False)
            )
            with amp_context:
                logits = model(inputs)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss_value = float(loss.detach().item())
            if not math.isfinite(loss_value):
                run_status = "diverged"
                divergence_reason = f"non-finite loss at step {step}"
                record = {
                    "step": step,
                    "wall_clock_sec": time.perf_counter() - start_time,
                    "train_loss": loss_value,
                    "lr": lr,
                    "event": "diverged",
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record), flush=True)
                break
            loss.backward()
            clip_gradients(raw_model.parameters(), max_grad_norm)
            optimizer.step()

            completed_step = step + 1
            completed_steps = completed_step
            step_seconds = time.perf_counter() - step_start
            record = {
                "step": completed_step,
                "wall_clock_sec": time.perf_counter() - start_time,
                "train_loss": float(loss.detach().item()),
                "lr": lr,
                "step_time_sec": step_seconds,
                "tokens_per_sec": batch_size * context_length / step_seconds,
            }
            if device.type == "cuda":
                record["max_memory_allocated_mb"] = torch.cuda.max_memory_allocated(device) / 1024**2
            if completed_step % eval_interval == 0 or completed_step == max_steps:
                last_val_loss = estimate_loss(
                    model,
                    val_data,
                    eval_batch_size,
                    context_length,
                    device,
                    batches=eval_batches,
                    amp_dtype=amp_dtype,
                )
                record["val_loss"] = last_val_loss
            log_file.write(json.dumps(record) + "\n")
            log_file.flush()
            print(json.dumps(record), flush=True)

            if completed_step % checkpoint_interval == 0 or completed_step == max_steps:
                save_checkpoint(
                    raw_model,
                    optimizer,
                    completed_step,
                    args.output_dir / f"checkpoint_{completed_step}.pt",
                )

    summary = {
        "status": run_status,
        "divergence_reason": divergence_reason,
        "final_step": completed_steps,
        "final_val_loss": last_val_loss,
        "total_wall_clock_sec": time.perf_counter() - start_time,
        "device": str(device),
        "precision": precision,
        "compile": bool(config.get("compile", False)),
        "model": model_config,
        "training": training,
        "optimizer": optimizer_config,
    }
    (args.output_dir / "summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
