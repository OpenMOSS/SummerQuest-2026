from __future__ import annotations

import argparse
import json
import math
import pickle
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping, softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW, get_batch, get_lr_cosine_schedule, save_checkpoint


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def evaluate(model: TransformerLM, data: np.ndarray, config: dict, device: str) -> float:
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(config["eval_batches"]):
            x, y = get_batch(data, config["batch_size"], config["context_length"], device)
            logits = model(x)
            losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1)).item())
    model.train()
    return float(sum(losses) / len(losses))


def sample_text(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    device: str,
) -> str:
    model.eval()
    ids = tokenizer.encode(prompt)
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
            logits = model(context)[0, -1] / max(temperature, 1e-6)
            probs = softmax(logits, dim=-1)
            sorted_probs, sorted_ids = torch.sort(probs, descending=True)
            cumulative = torch.cumsum(sorted_probs, dim=0)
            keep = cumulative <= top_p
            keep[0] = True
            filtered_probs = sorted_probs[keep]
            filtered_ids = sorted_ids[keep]
            filtered_probs = filtered_probs / filtered_probs.sum()
            next_id = filtered_ids[torch.multinomial(filtered_probs, 1).item()].item()
            ids.append(next_id)
            if tokenizer.decode([next_id]) == "<|endoftext|>":
                break
    model.train()
    return tokenizer.decode(ids)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    torch.manual_seed(config.get("seed", 0))
    np.random.seed(config.get("seed", 0))
    device = config.get("device", "cpu")
    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    train_data = np.load(config["train_tokens"], mmap_mode="r")
    valid_data = np.load(config["valid_tokens"], mmap_mode="r")
    with open(config["tokenizer"], "rb") as file:
        tokenizer_state = pickle.load(file)
    tokenizer = Tokenizer(
        tokenizer_state["vocab"],
        tokenizer_state["merges"],
        tokenizer_state.get("special_tokens", ["<|endoftext|>"]),
    )
    model = TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config.get("rope_theta", 10000.0),
        norm_mode=config.get("norm_mode", "pre"),
        use_rmsnorm=config.get("use_rmsnorm", True),
        use_rope=config.get("use_rope", True),
        ffn_type=config.get("ffn_type", "swiglu"),
    ).to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=config["max_lr"],
        weight_decay=config.get("weight_decay", 0.0),
        betas=tuple(config.get("betas", [0.9, 0.999])),
        eps=config.get("eps", 1e-8),
    )
    log_path = output_dir / config.get("log_name", "train.jsonl")
    start = time.time()
    final_val_loss = None
    status = "completed"
    divergence_step = None
    with open(log_path, "w", encoding="utf-8") as log_file:
        for step in range(config["steps"] + 1):
            lr = get_lr_cosine_schedule(
                step,
                config["max_lr"],
                config.get("min_lr", config["max_lr"] * 0.1),
                config.get("warmup_iters", 0),
                max(config.get("cosine_cycle_iters", config["steps"]), 1),
            )
            for group in optimizer.param_groups:
                group["lr"] = lr
            x, y = get_batch(train_data, config["batch_size"], config["context_length"], device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            optimizer.zero_grad()
            loss.backward()
            gradient_clipping(model.parameters(), config.get("grad_clip", 1.0))
            optimizer.step()
            if step % config.get("log_every", 10) == 0 or step == config["steps"]:
                val_loss = evaluate(model, valid_data, config, device)
                final_val_loss = val_loss
                record = {
                    "step": step,
                    "wall_clock_sec": time.time() - start,
                    "train_loss": float(loss.item()),
                    "val_loss": val_loss,
                    "lr": lr,
                }
                log_file.write(json.dumps(record) + "\n")
                log_file.flush()
                print(json.dumps(record))
                if not math.isfinite(record["train_loss"]) or not math.isfinite(record["val_loss"]):
                    status = "diverged_nonfinite_loss"
                    divergence_step = step
                    break
                divergence_threshold = config.get("divergence_val_loss_threshold")
                if (
                    divergence_threshold is not None
                    and step > 0
                    and record["val_loss"] >= divergence_threshold
                ):
                    status = "diverged_loss_explosion"
                    divergence_step = step
                    break
    checkpoint_path = output_dir / config.get("checkpoint_name", "checkpoint.pt")
    if status == "completed":
        save_checkpoint(model, optimizer, config["steps"], checkpoint_path)
        sample = sample_text(
            model,
            tokenizer,
            config.get("prompt", "Once upon a time"),
            config.get("max_new_tokens", 128),
            config.get("temperature", 0.8),
            config.get("top_p", 0.9),
            device,
        )
        checkpoint = str(checkpoint_path)
    else:
        sample = ""
        checkpoint = None
    summary = {
        "status": status,
        "divergence_step": divergence_step,
        "final_val_loss": final_val_loss,
        "total_training_time_sec": time.time() - start,
        "d_model": config["d_model"],
        "num_layers": config["num_layers"],
        "num_heads": config["num_heads"],
        "context_length": config["context_length"],
        "batch_size": config["batch_size"],
        "steps": config["steps"],
        "vocab_size": config["vocab_size"],
        "norm_mode": config.get("norm_mode", "pre"),
        "use_rmsnorm": config.get("use_rmsnorm", True),
        "use_rope": config.get("use_rope", True),
        "ffn_type": config.get("ffn_type", "swiglu"),
        "checkpoint": checkpoint,
        "sample": sample,
    }
    with open(output_dir / config.get("summary_name", "summary.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)
    with open(output_dir / config.get("sample_name", "sample.txt"), "w", encoding="utf-8") as file:
        file.write(sample)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
