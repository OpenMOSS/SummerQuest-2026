from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config_utils import load_config
from cs336_basics.training import AdamW, cross_entropy, get_batch
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Find the largest trainable batch size on the allocated GPU.")
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--batch-sizes", type=int, nargs="+", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    if not torch.cuda.is_available():
        raise RuntimeError("batch-size benchmark requires CUDA")
    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    dataset = np.memmap(config["train_path"], mode="r", dtype=np.dtype(config.get("data_dtype", "uint16")))
    model_config = config["model"]
    optimizer_config = config["optimizer"]
    context_length = int(model_config["context_length"])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as output:
        for batch_size in args.batch_sizes:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            record: dict[str, object] = {"batch_size": batch_size}
            try:
                model = TransformerLM(**model_config, device=device).to(device)
                optimizer = AdamW(model.parameters(), **optimizer_config)
                inputs, targets = get_batch(dataset, batch_size, context_length, device)
                start = time.perf_counter()
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(inputs)
                    loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
                loss.backward()
                optimizer.step()
                torch.cuda.synchronize()
                elapsed = time.perf_counter() - start
                record.update(
                    {
                        "status": "ok",
                        "loss": float(loss.detach().item()),
                        "step_time_sec": elapsed,
                        "tokens_per_sec": batch_size * context_length / elapsed,
                        "max_memory_allocated_mb": torch.cuda.max_memory_allocated(device) / 1024**2,
                    }
                )
                del loss, logits, inputs, targets, optimizer, model
            except torch.OutOfMemoryError as error:
                record.update({"status": "oom", "error": str(error)})
                output.write(json.dumps(record) + "\n")
                output.flush()
                print(json.dumps(record), flush=True)
                break
            output.write(json.dumps(record) + "\n")
            output.flush()
            print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main()
