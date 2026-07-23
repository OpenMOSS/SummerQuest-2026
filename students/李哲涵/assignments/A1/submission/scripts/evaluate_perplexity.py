from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import numpy as np
import torch

from _project_api import (
    build_model,
    build_optimizer,
    get_batch_fn as resolve_get_batch_fn,
    get_cross_entropy_fn as resolve_cross_entropy_fn,
    get_load_checkpoint_fn,
    load_config_for_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate validation loss and perplexity.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--valid-data", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--num-batches", type=int, required=True)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--output-path", type=Path, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    for path in (args.checkpoint, args.valid_data):
        if not path.is_file():
            raise FileNotFoundError(path)
    if args.batch_size <= 0 or args.num_batches <= 0:
        raise ValueError("--batch-size 和 --num-batches 必须大于 0")

    config = load_config_for_checkpoint(args.checkpoint)
    config["device"] = args.device
    device = torch.device(args.device)

    valid_data = np.load(args.valid_data, mmap_mode="r", allow_pickle=False)
    if valid_data.ndim != 1:
        raise ValueError("valid .npy 必须是一维 token id 数组")

    model = build_model(config).to(device)
    optimizer = build_optimizer(model, config)
    step = int(get_load_checkpoint_fn()(args.checkpoint, model, optimizer))
    model.eval()

    get_batch_fn = resolve_get_batch_fn()
    cross_entropy_fn = resolve_cross_entropy_fn()
    losses: list[float] = []
    with torch.no_grad():
        for _ in range(args.num_batches):
            inputs, targets = get_batch_fn(
                valid_data,
                args.batch_size,
                int(config["context_length"]),
                args.device,
            )
            logits = model(inputs)
            loss = cross_entropy_fn(
                logits.reshape(-1, logits.shape[-1]),
                targets.reshape(-1),
            )
            losses.append(float(loss.item()))

    average_loss = sum(losses) / len(losses)
    perplexity = math.exp(average_loss) if average_loss < 709 else math.inf
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_step": step,
        "valid_data": str(args.valid_data),
        "batch_size": args.batch_size,
        "num_batches": args.num_batches,
        "average_loss": average_loss,
        "perplexity": perplexity,
    }

    output_path = args.output_path or (
        args.checkpoint.parent / f"{args.checkpoint.stem}_perplexity.json"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as file:
        json.dump(result, file, ensure_ascii=False, indent=2)
        file.write("\n")

    print(f"checkpoint_step={step}")
    print(f"average_loss={average_loss:.6f}")
    print(f"perplexity={perplexity:.6f}")
    print(f"saved: {output_path}")


if __name__ == "__main__":
    main()
