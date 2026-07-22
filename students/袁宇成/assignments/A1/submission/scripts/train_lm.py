#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from cs336_basics.model import TransformerLM
from cs336_basics.training import AdamW, clip_gradients, cosine_schedule, cross_entropy, get_batch, load_checkpoint, save_checkpoint


def read_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def autocast_context(device: torch.device, amp_dtype: str | None):
    if device.type == "cuda" and amp_dtype:
        dtype = {"bfloat16": torch.bfloat16, "float16": torch.float16}[amp_dtype]
        return torch.autocast(device_type="cuda", dtype=dtype)
    from contextlib import nullcontext

    return nullcontext()


@torch.no_grad()
def validation_loss(model, data, config, device, batch_size: int, world_size: int) -> float:
    model.eval()
    losses = []
    for _ in range(config["val_batches"]):
        x, y = get_batch(data, batch_size, config["context_length"], str(device))
        with autocast_context(device, config.get("amp_dtype")):
            logits = model(x)
            losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    loss = torch.tensor(sum(losses) / len(losses), device=device)
    if world_size > 1:
        dist.all_reduce(loss, op=dist.ReduceOp.SUM)
        loss /= world_size
    model.train()
    return loss.item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the assignment Transformer language model.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--val-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    config = read_config(args.config)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size > 8:
        raise ValueError("A1 training is limited to at most 8 devices")
    if world_size > 1:
        dist.init_process_group(backend="nccl")
        torch.cuda.set_device(local_rank)
        args.device = f"cuda:{local_rank}"

    seed = config.get("seed", 42) + rank
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    device = torch.device(args.device)
    if config["batch_size"] % world_size:
        raise ValueError("global batch_size must be divisible by world_size")
    local_batch_size = config["batch_size"] // world_size
    dtype = np.dtype(config.get("data_dtype", "uint16"))
    train_data = np.memmap(args.train_data, dtype=dtype, mode="r")
    val_data = np.memmap(args.val_data, dtype=dtype, mode="r")

    ablation = config.get("ablation", "baseline")
    ablations = {
        "use_rmsnorm": ablation != "no_rmsnorm",
        "post_norm": ablation == "post_norm",
        "use_rope": ablation != "no_rope",
        "ffn_variant": "silu" if ablation == "silu_ffn" else "swiglu",
        "silu_d_ff": config.get("silu_d_ff"),
    }
    model = TransformerLM(
        config["vocab_size"],
        config["context_length"],
        config["d_model"],
        config["num_layers"],
        config["num_heads"],
        config["d_ff"],
        config.get("rope_theta", 10000.0),
        **ablations,
    ).to(device)
    train_model = DistributedDataParallel(model, device_ids=[local_rank]) if world_size > 1 else model
    optimizer = AdamW(
        model.parameters(),
        lr=config["max_lr"],
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("eps", 1e-8),
        weight_decay=config.get("weight_decay", 0.1),
    )
    start_step = load_checkpoint(args.resume, model, optimizer) if args.resume else 0
    output_dir = Path(args.output_dir)
    if rank == 0:
        output_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    log_path = output_dir / "metrics.jsonl"
    start_time = time.perf_counter()
    last_val_loss = None

    log_file = open(log_path, "a", encoding="utf-8") if rank == 0 else None
    try:
        for step in range(start_step, config["training_steps"]):
            lr = cosine_schedule(
                step,
                config["max_lr"],
                config["min_lr"],
                config["warmup_steps"],
                max(config["training_steps"] - 1, config["warmup_steps"]),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_data, local_batch_size, config["context_length"], str(device))
            optimizer.zero_grad(set_to_none=True)
            with autocast_context(device, config.get("amp_dtype")):
                logits = train_model(x)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            clip_gradients(model.parameters(), config.get("max_grad_norm", 1.0))
            optimizer.step()

            current_step = step + 1
            log_due = current_step == 1 or current_step % config["log_interval"] == 0
            mean_train_loss = loss.detach()
            if log_due and world_size > 1:
                dist.all_reduce(mean_train_loss, op=dist.ReduceOp.SUM)
                mean_train_loss /= world_size
            if rank == 0 and log_due:
                record = {
                    "step": current_step,
                    "wall_clock_sec": time.perf_counter() - start_time,
                    "processed_tokens": current_step * config["batch_size"] * config["context_length"],
                    "train_loss": mean_train_loss.item(),
                    "lr": lr,
                    "world_size": world_size,
                    "peak_memory_gib": torch.cuda.max_memory_allocated(device) / 2**30 if device.type == "cuda" else 0,
                }
                if current_step % config["val_interval"] == 0 or current_step == config["training_steps"]:
                    last_val_loss = validation_loss(
                        train_model, val_data, config, device, local_batch_size, world_size
                    )
                    record["val_loss"] = last_val_loss
                print(json.dumps(record), flush=True)
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
            elif world_size > 1 and (
                current_step % config["val_interval"] == 0 or current_step == config["training_steps"]
            ):
                last_val_loss = validation_loss(train_model, val_data, config, device, local_batch_size, world_size)
            if rank == 0 and (
                current_step % config["checkpoint_interval"] == 0 or current_step == config["training_steps"]
            ):
                save_checkpoint(model, optimizer, current_step, output_dir / "checkpoint.pt")
    finally:
        if log_file is not None:
            log_file.close()

    # Use a separate validation sample for the summary instead of copying the
    # final periodic point from the training log.
    final_val = validation_loss(train_model, val_data, config, device, local_batch_size, world_size)
    summary = {
        "final_val_loss": final_val,
        "total_training_time_sec": time.perf_counter() - start_time,
        "parameter_count": sum(parameter.numel() for parameter in model.parameters()),
        "config": config,
    }
    if rank == 0:
        (output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
        print(json.dumps(summary, indent=2))
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
