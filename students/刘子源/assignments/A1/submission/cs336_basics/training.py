# 训练主流程都在这儿，配置项很多但说到底就是取 batch、算 loss、存盘。

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

from .data import get_batch
from .model import TransformerLM
from .nn_utils import cross_entropy, gradient_clipping
from .optimizer import AdamW, get_lr_cosine_schedule
from .serialization import load_checkpoint, save_checkpoint


def load_token_array(path: str | Path, dtype: str = "uint16") -> np.ndarray:
    path = Path(path)
    if path.suffix == ".npy":
        return np.load(path, mmap_mode="r")
    return np.memmap(path, mode="r", dtype=np.dtype(dtype))


def build_model(config: dict, device: str) -> TransformerLM:
    return TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10_000.0),
        use_rmsnorm=config.get("use_rmsnorm", True),
        post_norm=config.get("post_norm", False),
        use_rope=config.get("use_rope", True),
        ffn_kind=config.get("ffn_kind", "swiglu"),
        device=device,
    )


def _last_logged_wall_clock(log_path: Path) -> float:
    if not log_path.exists():
        return 0.0
    last_one = None
    with log_path.open(encoding="utf-8") as log_file:
        for line in log_file:
            if line.strip():
                last_one = json.loads(line)
    return float(last_one.get("wall_clock_sec", 0.0)) if last_one else 0.0


@torch.no_grad()
def evaluate(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    old_is_train = model.training
    model.eval()
    loss_list = []
    try:
        for _ in range(eval_batches):
            batch_x, batch_y = get_batch(dataset, batch_size, context_length, device)
            logits = model(batch_x)
            loss_list.append(
                cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1)).item()
            )
    finally:
        if old_is_train:
            model.train()
    return float(sum(loss_list) / len(loss_list))


def train(config: dict) -> dict:
    seed = int(config.get("seed", 0))
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    device = config.get("device", "cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / config.get("log_filename", "train.jsonl")
    summary_path = output_dir / config.get("summary_filename", "summary.json")
    checkpoint_path = output_dir / config.get("checkpoint_filename", "checkpoint.pt")

    train_data = load_token_array(config["train_data"], config.get("data_dtype", "uint16"))
    validation_data = load_token_array(config["validation_data"], config.get("data_dtype", "uint16"))
    model = build_model(config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["max_learning_rate"],
        betas=tuple(config.get("betas", [0.9, 0.95])),
        eps=config.get("adam_eps", 1e-8),
        weight_decay=config.get("weight_decay", 0.1),
    )

    start_step = 0
    resume_path = config.get("resume_from")
    if resume_path:
        start_step = load_checkpoint(resume_path, model, optimizer)
    elapsed_offset = _last_logged_wall_clock(log_path) if resume_path else 0.0

    total_steps = int(config["total_steps"])
    context_length = int(config["context_length"])
    batch_size = int(config["batch_size"])
    log_interval = int(config.get("log_interval", 10))
    eval_interval = int(config.get("eval_interval", 100))
    checkpoint_interval = int(config.get("checkpoint_interval", 1000))
    eval_batches = int(config.get("eval_batches", 20))
    max_grad_norm = float(config.get("max_grad_norm", 1.0))
    started = time.perf_counter()
    final_val_loss = None

    model.train()
    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(start_step, total_steps):
            learning_rate = get_lr_cosine_schedule(
                step,
                config["max_learning_rate"],
                config["min_learning_rate"],
                config["warmup_steps"],
                config.get("cosine_cycle_steps", total_steps),
            )
            for param_group in optimizer.param_groups:
                param_group["lr"] = learning_rate

            batch_x, batch_y = get_batch(train_data, batch_size, context_length, device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(batch_x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), batch_y.reshape(-1))
            loss.backward()
            gradient_clipping(model.parameters(), max_grad_norm)
            optimizer.step()

            need_eval = step % eval_interval == 0 or step == total_steps - 1
            if need_eval:
                final_val_loss = evaluate(
                    model, validation_data, batch_size, context_length, device, eval_batches
                )
            if step % log_interval == 0 or need_eval:
                one_log = {
                    "step": step,
                    "wall_clock_sec": elapsed_offset + time.perf_counter() - started,
                    "processed_tokens": (step + 1) * batch_size * context_length,
                    "train_loss": float(loss.item()),
                    "lr": learning_rate,
                }
                if need_eval:
                    assert final_val_loss is not None
                    one_log["val_loss"] = final_val_loss
                # 每行单独一个 JSON，训练半路挂了前面的记录还能捞。
                log_file.write(json.dumps(one_log, ensure_ascii=False) + "\n")
                log_file.flush()
            if checkpoint_interval > 0 and (step + 1) % checkpoint_interval == 0:
                save_checkpoint(model, optimizer, step + 1, checkpoint_path)

    save_checkpoint(model, optimizer, total_steps, checkpoint_path)
    summary = {
        "run_name": config.get("run_name", output_dir.name),
        "dataset": config.get("dataset", "unspecified"),
        "final_val_loss": final_val_loss,
        "total_training_time_sec": elapsed_offset + time.perf_counter() - started,
        "config": {
            key: config[key]
            for key in (
                "vocab_size",
                "context_length",
                "d_model",
                "d_ff",
                "num_layers",
                "num_heads",
                "batch_size",
                "total_steps",
            )
        },
        "checkpoint": str(checkpoint_path),
        "log": str(log_path),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary
