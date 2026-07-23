from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy, gradient_clipping


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/tinystories_small.json")
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 64, 128, 256, 512, 1024, 2048, 4096, 8192])
    parser.add_argument("--out", default="runs/batch_memory_probe/summary.json")
    return parser.parse_args()


def load_config(path: str) -> dict:
    with open(path, encoding="utf-8") as file:
        return json.load(file)


def main() -> int:
    args = parse_args()
    config = load_config(args.config)
    device = config.get("device", "cuda")
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    results = []
    for batch_size in args.batch_sizes:
        record = {"batch_size": batch_size, "status": "unknown"}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            start = time.time()
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
            x = torch.randint(
                0,
                config["vocab_size"],
                (batch_size, config["context_length"]),
                dtype=torch.long,
                device=device,
            )
            y = torch.randint(
                0,
                config["vocab_size"],
                (batch_size, config["context_length"]),
                dtype=torch.long,
                device=device,
            )
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            gradient_clipping(model.parameters(), config.get("grad_clip", 1.0))
            torch.cuda.synchronize()
            record.update(
                {
                    "status": "ok",
                    "loss": float(loss.item()),
                    "elapsed_sec": time.time() - start,
                    "peak_memory_gib": torch.cuda.max_memory_allocated() / 1024**3,
                }
            )
            del model, x, y, logits, loss
        except torch.cuda.OutOfMemoryError as error:
            record.update({"status": "oom", "error": str(error).splitlines()[0]})
            results.append(record)
            print(json.dumps(record), flush=True)
            break
        except RuntimeError as error:
            if "out of memory" not in str(error).lower():
                raise
            record.update({"status": "oom", "error": str(error).splitlines()[0]})
            results.append(record)
            print(json.dumps(record), flush=True)
            break
        results.append(record)
        print(json.dumps(record), flush=True)

    ok_batches = [item["batch_size"] for item in results if item["status"] == "ok"]
    summary = {
        "config": args.config,
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else device,
        "tested_batch_sizes": args.batch_sizes,
        "max_successful_batch_size": max(ok_batches) if ok_batches else None,
        "first_oom_batch_size": next((item["batch_size"] for item in results if item["status"] == "oom"), None),
        "results": results,
    }
    out_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
