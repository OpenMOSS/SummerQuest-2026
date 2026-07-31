from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch


def accumulation_experiment(n=1000, value=0.1, device="cpu"):
    x32 = torch.full((n,), value, dtype=torch.float32, device=device)
    x16 = x32.half()
    return {
        "fp32_input_fp32_accumulator": x32.sum(dtype=torch.float32).item(),
        "fp32_input_fp16_accumulator": x32.sum(dtype=torch.float16).item(),
        "fp16_input_fp32_accumulator": x16.sum(dtype=torch.float32).item(),
        "fp16_input_fp16_accumulator": x16.sum(dtype=torch.float16).item(),
        "reference_float64": torch.full((n,), value, dtype=torch.float64, device=device).sum().item(),
    }


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=1000)
    p.add_argument("--value", type=float, default=0.1)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--output", default="results/mixed_precision_accumulation.json")
    args = p.parse_args()
    result = accumulation_experiment(args.n, args.value, args.device)
    path = Path(args.output)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
