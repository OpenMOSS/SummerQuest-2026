from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optimizer import AdamW
from cs336_basics.training import (
    clip_gradients,
    cosine_learning_rate,
    cross_entropy,
    get_batch,
    load_checkpoint,
    save_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a decoder-only Transformer language model.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--device", default=None)
    parser.add_argument("--resume", type=Path, default=None)
    return parser.parse_args()


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def build_model(config: dict, device: torch.device) -> TransformerLM:
    return TransformerLM(
        vocab_size=config["vocab_size"],
        context_length=config["context_length"],
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=config["rope_theta"],
        remove_rmsnorm=config.get("remove_rmsnorm", False),
        use_post_norm=config.get("use_post_norm", False),
        remove_rope=config.get("remove_rope", False),
        ffn_type=config.get("ffn_type", "swiglu"),
        device=device,
    )


@torch.no_grad()
def validation_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    batches: int,
    device: torch.device,
) -> float:
    model.eval()
    losses = []
    for _ in range(batches):
        inputs, targets = get_batch(dataset, batch_size, context_length, device)
        logits = model(inputs)
        losses.append(cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1)).item())
    model.train()
    return float(np.mean(losses))


def main() -> None:
    args = parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    model_config = config["model"]
    training = config["training"]
    data = config["data"]
    output = config["output"]

    device = resolve_device(args.device or training.get("device", "auto"))
    seed = training.get("seed", 42)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(seed)

    train_data = np.load(data["train"], mmap_mode="r")
    validation_data = np.load(data["validation"], mmap_mode="r")
    model = build_model(model_config, device)
    optimizer = AdamW(
        model.parameters(),
        lr=training["max_learning_rate"],
        betas=tuple(training.get("betas", [0.9, 0.95])),
        eps=training.get("eps", 1e-8),
        weight_decay=training.get("weight_decay", 0.1),
    )

    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)

    log_path = Path(output["log"])
    checkpoint_dir = Path(output["checkpoint_dir"])
    log_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    model.train()

    with log_path.open("a", encoding="utf-8") as log_file:
        for step in range(start_step, training["max_steps"]):
            learning_rate = cosine_learning_rate(
                step,
                training["max_learning_rate"],
                training["min_learning_rate"],
                training["warmup_steps"],
                training["max_steps"],
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
            logits = model(inputs)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            clip_gradients(model.parameters(), training["max_grad_norm"])
            optimizer.step()

            completed_step = step + 1
            should_log = completed_step % training["log_interval"] == 0 or completed_step == 1
            should_evaluate = completed_step % training["eval_interval"] == 0
            if should_log or should_evaluate:
                record = {
                    "step": completed_step,
                    "tokens": completed_step * training["batch_size"] * model_config["context_length"],
                    "wall_time_seconds": time.perf_counter() - started,
                    "learning_rate": learning_rate,
                    "train_loss": loss.item(),
                }
                if should_evaluate:
                    record["validation_loss"] = validation_loss(
                        model,
                        validation_data,
                        training["eval_batch_size"],
                        model_config["context_length"],
                        training["eval_batches"],
                        device,
                    )
                line = json.dumps(record, sort_keys=True)
                print(line, flush=True)
                log_file.write(line + "\n")
                log_file.flush()

            if completed_step % training["checkpoint_interval"] == 0:
                save_checkpoint(
                    model,
                    optimizer,
                    completed_step,
                    checkpoint_dir / f"step-{completed_step:07d}.pt",
                )

    save_checkpoint(model, optimizer, training["max_steps"], checkpoint_dir / "final.pt")


if __name__ == "__main__":
    main()
