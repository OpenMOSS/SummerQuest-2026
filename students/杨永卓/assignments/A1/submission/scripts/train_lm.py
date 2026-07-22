#!/usr/bin/env python3
"""Train a decoder-only Transformer and emit JSONL evidence logs."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from cs336_basics.experiment import build_model, count_parameters, load_json, resolve_dtype, save_json, set_seed
from cs336_basics.training import AdamW, clip_gradients, cross_entropy, get_batch, get_lr_cosine_schedule


def autocast_context(device: torch.device, dtype: torch.dtype):
    if device.type == "cuda" and dtype != torch.float32:
        return torch.autocast(device_type="cuda", dtype=dtype)
    return nullcontext()


@torch.no_grad()
def evaluate(model, dataset, config, device, amp_dtype) -> float:
    model.eval()
    losses = []
    for _ in range(config["training"]["val_batches"]):
        x, y = get_batch(
            dataset,
            config["training"]["batch_size"],
            config["model"]["context_length"],
            device,
        )
        with autocast_context(device, amp_dtype):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        losses.append(loss.float())
    model.train()
    return torch.stack(losses).mean().item()


def append_log(path: Path, record: dict) -> None:
    with path.open("a", encoding="utf-8") as output:
        output.write(json.dumps(record, sort_keys=True) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    config = load_json(args.config)
    set_seed(config.get("seed", 42))
    if not torch.cuda.is_available() and config.get("device", "cuda") == "cuda":
        raise RuntimeError("CUDA is required by this configuration")
    device = torch.device(config.get("device", "cuda"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.set_float32_matmul_precision("high")

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    save_json(output_dir / "config.json", config)
    log_path = output_dir / "train.jsonl"
    summary_path = output_dir / "summary.json"
    checkpoint_path = output_dir / "checkpoint_last.pt"
    wall_clock_offset = 0.0
    if log_path.exists():
        with log_path.open(encoding="utf-8") as existing_log:
            existing_records = [json.loads(line) for line in existing_log if line.strip()]
        if existing_records:
            wall_clock_offset = float(existing_records[-1].get("wall_clock_sec", 0.0))

    train_data = np.memmap(config["data"]["train"], dtype=np.uint16, mode="r")
    val_data = np.memmap(config["data"]["validation"], dtype=np.uint16, mode="r")
    model = build_model(config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["training"]["max_lr"],
        betas=tuple(config["training"].get("betas", [0.9, 0.95])),
        eps=config["training"].get("eps", 1e-8),
        weight_decay=config["training"].get("weight_decay", 0.1),
    )
    start_step = 0
    if args.resume and checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = int(checkpoint["iteration"])

    train_model = model
    if config["training"].get("compile", False):
        train_model = torch.compile(model)
    amp_dtype = resolve_dtype(config["training"].get("amp_dtype", "bfloat16"))
    total_steps = config["training"]["steps"]
    tokens_per_step = config["training"]["batch_size"] * config["model"]["context_length"]
    started = time.perf_counter()
    best_val_loss = float("inf")
    final_train_loss = float("nan")

    if start_step == 0:
        initial_val = evaluate(train_model, val_data, config, device, amp_dtype)
        best_val_loss = initial_val
        append_log(log_path, {"step": 0, "wall_clock_sec": 0.0, "val_loss": initial_val, "lr": 0.0})

    for iteration in range(start_step, total_steps):
        lr = get_lr_cosine_schedule(
            iteration,
            config["training"]["max_lr"],
            config["training"]["min_lr"],
            config["training"]["warmup_steps"],
            config["training"]["cosine_steps"],
        )
        for group in optimizer.param_groups:
            group["lr"] = lr
        x, y = get_batch(
            train_data,
            config["training"]["batch_size"],
            config["model"]["context_length"],
            device,
        )
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, amp_dtype):
            logits = train_model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        loss.backward()
        clip_gradients(model.parameters(), config["training"].get("grad_clip", 1.0))
        optimizer.step()
        step = iteration + 1
        final_train_loss = loss.item()
        should_validate = step % config["training"]["val_interval"] == 0 or step == total_steps
        should_log = step % config["training"]["log_interval"] == 0 or should_validate
        record = None
        if should_log:
            record = {
                "step": step,
                "wall_clock_sec": wall_clock_offset + time.perf_counter() - started,
                "train_loss": final_train_loss,
                "lr": lr,
                "processed_tokens": step * tokens_per_step,
            }
        if should_validate:
            val_loss = evaluate(train_model, val_data, config, device, amp_dtype)
            best_val_loss = min(best_val_loss, val_loss)
            record["val_loss"] = val_loss
        if record is not None:
            append_log(log_path, record)
            print(json.dumps(record, sort_keys=True), flush=True)
        checkpoint_interval = config["training"].get("checkpoint_interval", total_steps)
        if step % checkpoint_interval == 0 or step == total_steps:
            torch.save(
                {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": step},
                checkpoint_path,
            )

    total_seconds = wall_clock_offset + time.perf_counter() - started
    final_val_loss = evaluate(train_model, val_data, config, device, amp_dtype)
    summary = {
        "run_name": config["run_name"],
        "model": config["model"],
        "ablation": config.get("ablation", {}),
        "parameter_count": count_parameters(model),
        "batch_size": config["training"]["batch_size"],
        "steps": total_steps,
        "context_length": config["model"]["context_length"],
        "processed_tokens": total_steps * tokens_per_step,
        "final_train_loss": final_train_loss,
        "final_val_loss": final_val_loss,
        "best_val_loss": min(best_val_loss, final_val_loss),
        "total_training_sec": total_seconds,
        "tokens_per_sec": total_steps * tokens_per_step / total_seconds,
        "max_cuda_memory_bytes": torch.cuda.max_memory_allocated(device) if device.type == "cuda" else 0,
    }
    save_json(summary_path, summary)
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
