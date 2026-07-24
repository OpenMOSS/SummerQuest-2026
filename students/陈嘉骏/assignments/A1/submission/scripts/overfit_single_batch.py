from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    AdamW,
    clip_gradients,
    cross_entropy,
    get_batch,
    load_token_dataset,
    save_checkpoint,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Overfit a Transformer LM on one fixed batch.")
    parser.add_argument("--config", default="configs/tinystories_debug.json")
    parser.add_argument("--output-dir", default="runs/tinystories_overfit")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--log-interval", type=int, default=50)
    parser.add_argument("--seed", type=int, default=42)
    return parser


def _write_json(path: Path, value: object) -> None:
    temporary_path = path.with_name(path.name + ".tmp")
    with open(temporary_path, "w", encoding="utf-8") as f:
        json.dump(value, f, ensure_ascii=False, indent=2)
        f.write("\n")
    temporary_path.replace(path)


def main() -> None:
    args = _build_parser().parse_args()
    if args.steps <= 0 or args.learning_rate <= 0 or args.max_grad_norm <= 0 or args.log_interval <= 0:
        raise ValueError("steps, learning-rate, max-grad-norm, and log-interval must be positive.")

    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)

    training_configuration = configuration["training"]
    device = torch.device(training_configuration["device"])
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "overfit.jsonl"
    log_path.unlink(missing_ok=True)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TransformerLM(**configuration["model"]).to(device)
    dataset = load_token_dataset(configuration["data"]["train"])
    inputs, targets = get_batch(
        dataset=dataset,
        batch_size=training_configuration["batch_size"],
        context_length=training_configuration["context_length"],
        device=device,
    )
    optimizer = AdamW(
        model.parameters(),
        lr=args.learning_rate,
        betas=(training_configuration["beta1"], training_configuration["beta2"]),
        eps=training_configuration["eps"],
        weight_decay=0.0,
    )

    start_time = time.perf_counter()
    initial_loss: float | None = None
    model.train()
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = cross_entropy(logits, targets)
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite loss at step {step}: {float(loss)}")
        if initial_loss is None:
            initial_loss = float(loss.detach())

        loss.backward()
        clip_gradients(model.parameters(), args.max_grad_norm)
        optimizer.step()

        if step == 1 or step % args.log_interval == 0 or step == args.steps:
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            record = {
                "step": step,
                "wall_clock_sec": time.perf_counter() - start_time,
                "train_loss": float(loss.detach()),
            }
            with open(log_path, "a", encoding="utf-8") as f:
                json.dump(record, f)
                f.write("\n")
            print(json.dumps(record))

    model.eval()
    with torch.no_grad():
        final_loss = float(cross_entropy(model(inputs), targets))
    assert initial_loss is not None

    checkpoint_path = output_dir / "checkpoint_final.pt"
    save_checkpoint(model, optimizer, args.steps, checkpoint_path)
    summary = {
        "config": args.config,
        "device": str(device),
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": int(inputs.shape[0]),
        "context_length": int(inputs.shape[1]),
        "learning_rate": args.learning_rate,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_reduction": initial_loss - final_loss,
        "elapsed_seconds": time.perf_counter() - start_time,
        "checkpoint_path": str(checkpoint_path),
        "passed": math.isfinite(final_loss) and final_loss < 0.1,
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))

    if final_loss >= initial_loss:
        raise RuntimeError("Loss did not decrease on the fixed batch.")


if __name__ == "__main__":
    main()
