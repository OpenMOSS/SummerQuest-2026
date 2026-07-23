from __future__ import annotations

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import torch
from torch import nn


DTYPES = {
    "fp16": torch.float16,
    "bf16": torch.bfloat16,
}


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        x = self.fc2(x)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed precision probes for CS336 Assignment 2 Section 2.1.5.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=DTYPES.keys(), default="fp16")
    parser.add_argument("--in-features", type=int, default=32)
    parser.add_argument("--out-features", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("results/mixed_precision/toy_dtype.json"))
    return parser.parse_args()


def accumulation_probe() -> dict[str, str]:
    results: dict[str, str] = {}

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    results["fp32_accumulator_fp32_input"] = str(s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp16_accumulator_fp16_input"] = str(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp32_accumulator_fp16_input"] = str(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    results["fp32_accumulator_casted_fp16_input"] = str(s)
    return results


def autocast_context(device: torch.device, dtype_name: str):
    if device.type != "cuda":
        return nullcontext()
    return torch.autocast(device_type="cuda", dtype=DTYPES[dtype_name])


def toy_dtype_probe(args: argparse.Namespace) -> dict[str, str]:
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        device = torch.device("cuda", 0 if device.index is None else device.index)
        torch.cuda.set_device(device.index)

    model = ToyModel(args.in_features, args.out_features).to(device)
    x = torch.randn(args.batch_size, args.in_features, device=device)
    targets = torch.randint(0, args.out_features, (args.batch_size,), device=device)

    observed: dict[str, str] = {
        "parameter": str(next(model.parameters()).dtype),
    }

    def record(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            observed[name] = str(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(record("fc1_output")),
        model.ln.register_forward_hook(record("layernorm_output")),
        model.fc2.register_forward_hook(record("logits")),
    ]
    try:
        with autocast_context(device, args.dtype):
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    observed["loss"] = str(loss.dtype)
    observed["gradient"] = str(next(model.parameters()).grad.dtype)
    return observed


def main() -> int:
    args = parse_args()
    payload = {
        "config": {
            "device": args.device,
            "dtype": args.dtype,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "batch_size": args.batch_size,
            "seed": args.seed,
        },
        "accumulation": accumulation_probe(),
        "toy_model_dtypes": toy_dtype_probe(args),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
