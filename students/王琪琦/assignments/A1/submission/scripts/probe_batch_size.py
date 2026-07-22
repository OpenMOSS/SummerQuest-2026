from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradients, cross_entropy
from cs336_basics.optimizer import AdamW


def main() -> None:
    parser = argparse.ArgumentParser(description="Probe train-step memory for candidate batch sizes.")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[128, 256, 512, 1024])
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    model_config = config["model"]
    training = config["training"]
    device = torch.device(args.device)
    model = TransformerLM(**model_config, device=device)
    optimizer = AdamW(
        model.parameters(),
        lr=training["max_learning_rate"],
        betas=tuple(training["betas"]),
        eps=training["eps"],
        weight_decay=training["weight_decay"],
    )
    results = []
    for batch_size in args.batches:
        torch.cuda.reset_peak_memory_stats(device)
        try:
            tokens = torch.randint(
                model_config["vocab_size"],
                (batch_size, model_config["context_length"] + 1),
                device=device,
            )
            optimizer.zero_grad(set_to_none=True)
            loss = cross_entropy(model(tokens[:, :-1]), tokens[:, 1:])
            loss.backward()
            clip_gradients(model.parameters(), training["max_grad_norm"])
            optimizer.step()
            peak_bytes = torch.cuda.max_memory_allocated(device)
            result = {
                "batch_size": batch_size,
                "status": "passed",
                "peak_allocated_gib": peak_bytes / 1024**3,
            }
            del tokens, loss
        except torch.OutOfMemoryError as error:
            result = {"batch_size": batch_size, "status": "oom", "error": str(error)}
            optimizer.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
            results.append(result)
            print(json.dumps(result))
            break
        results.append(result)
        print(json.dumps(result))

    report = {
        "gpu": torch.cuda.get_device_name(device),
        "total_memory_gib": torch.cuda.get_device_properties(device).total_memory / 1024**3,
        "context_length": model_config["context_length"],
        "results": results,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
