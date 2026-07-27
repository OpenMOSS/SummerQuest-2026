from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .benchmark import build_parser as benchmark_parser
from .benchmark import run_benchmark
from .common import command_string, software_metadata, utc_now, write_json


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.ln(x)
        return self.fc2(x)


def accumulation_experiment() -> list[dict[str, Any]]:
    results = []

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    results.append({"name": "fp32_input_fp32_accumulator", "value": float(s), "dtype": str(s.dtype)})

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results.append({"name": "fp16_input_fp16_accumulator", "value": float(s), "dtype": str(s.dtype)})

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results.append({"name": "fp16_input_implicit_fp32_accumulator", "value": float(s), "dtype": str(s.dtype)})

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    results.append({"name": "fp16_input_explicit_fp32_accumulator", "value": float(s), "dtype": str(s.dtype)})
    return results


def toy_dtype_experiment(device: torch.device) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("ToyModel dtype experiment requires CUDA BF16 autocast")
    torch.manual_seed(1337)
    model = ToyModel(32, 7).to(device)
    x = torch.randn(16, 32, device=device)
    target = torch.randint(0, 7, (16,), device=device)
    observed: dict[str, str] = {"parameter": str(next(model.parameters()).dtype)}

    hooks = [
        model.fc1.register_forward_hook(lambda _m, _i, o: observed.__setitem__("fc1_output", str(o.dtype))),
        model.ln.register_forward_hook(lambda _m, _i, o: observed.__setitem__("layernorm_output", str(o.dtype))),
    ]
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        observed["logits"] = str(logits.dtype)
        loss = nn.functional.cross_entropy(logits.float(), target)
        observed["loss"] = str(loss.dtype)
    loss.backward()
    gradient = next(model.parameters()).grad
    if gradient is None:
        raise RuntimeError("ToyModel did not produce a gradient")
    observed["gradient"] = str(gradient.dtype)
    for hook in hooks:
        hook.remove()
    return {"dtypes": observed, "loss": float(loss.detach())}


def build_parser() -> argparse.ArgumentParser:
    parser = benchmark_parser()
    parser.description = "Run A2-P accumulation, ToyModel, and FP32/BF16 benchmark comparisons"
    parser.set_defaults(mode="train_step", model_size="small", batch_size=4, context_length=512, warmup=5, steps=10)
    parser.add_argument("--combined-output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    device = torch.device(args.device)
    base_output = args.output
    benchmark_results = {}
    for dtype in ("fp32", "bf16"):
        args.dtype = dtype
        args.output = base_output.with_name(f"{base_output.stem}_{dtype}{base_output.suffix}")
        benchmark_results[dtype] = run_benchmark(args)

    payload = {
        "schema_version": 1,
        "timestamp_utc": utc_now(),
        "command": command_string(),
        "result_file": args.combined_output.name,
        "accumulation": accumulation_experiment(),
        "toy_model": toy_dtype_experiment(device),
        "benchmark": benchmark_results,
        "software": software_metadata(device),
    }
    write_json(args.combined_output, payload)
    print(f"wrote {args.combined_output}")


if __name__ == "__main__":
    main()
