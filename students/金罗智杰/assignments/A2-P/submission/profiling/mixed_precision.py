from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


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
    parser = argparse.ArgumentParser(description="Run the A2 mixed-precision accumulation and ToyModel experiments.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    if args.output.exists() and args.output.is_dir():
        parser.error("--output must be a JSON file path, not a directory")
    if args.output.suffix.lower() != ".json":
        parser.error("--output must end with .json")
    return args


def run_accumulation_experiment() -> list[dict[str, Any]]:
    cases: list[dict[str, Any]] = []

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float32)
    cases.append(
        {
            "name": "fp32_accumulator_fp32_input",
            "accumulator_dtype": str(accumulator.dtype),
            "input_dtype": str(torch.float32),
            "explicit_input_cast": False,
            "result": accumulator.item(),
            "absolute_error_from_10": abs(accumulator.item() - 10.0),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    cases.append(
        {
            "name": "fp16_accumulator_fp16_input",
            "accumulator_dtype": str(accumulator.dtype),
            "input_dtype": str(torch.float16),
            "explicit_input_cast": False,
            "result": accumulator.item(),
            "absolute_error_from_10": abs(accumulator.item() - 10.0),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        accumulator += torch.tensor(0.01, dtype=torch.float16)
    cases.append(
        {
            "name": "fp32_accumulator_fp16_input",
            "accumulator_dtype": str(accumulator.dtype),
            "input_dtype": str(torch.float16),
            "explicit_input_cast": False,
            "result": accumulator.item(),
            "absolute_error_from_10": abs(accumulator.item() - 10.0),
        }
    )

    accumulator = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        value = torch.tensor(0.01, dtype=torch.float16)
        accumulator += value.type(torch.float32)
    cases.append(
        {
            "name": "fp32_accumulator_explicit_fp16_to_fp32_input",
            "accumulator_dtype": str(accumulator.dtype),
            "input_dtype": str(torch.float16),
            "explicit_input_cast": True,
            "result": accumulator.item(),
            "absolute_error_from_10": abs(accumulator.item() - 10.0),
        }
    )

    return cases


def run_toy_model_experiment(device: torch.device, seed: int) -> dict[str, Any]:
    if device.type != "cuda":
        raise RuntimeError("The ToyModel autocast experiment must run on a CUDA GPU.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available in this environment.")

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    in_features = 16
    out_features = 8
    batch_size = 4
    model = ToyModel(in_features, out_features).to(device)
    model.train()

    activations: dict[str, str] = {}

    def capture_dtype(name: str):
        def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
            activations[name] = str(output.dtype)

        return hook

    handles = [
        model.fc1.register_forward_hook(capture_dtype("fc1_output")),
        model.ln.register_forward_hook(capture_dtype("layer_norm_output")),
    ]

    inputs = torch.randn(batch_size, in_features, device=device, dtype=torch.float32)
    targets = torch.randint(0, out_features, (batch_size,), device=device)

    try:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            parameter_dtype_inside_autocast = str(next(model.parameters()).dtype)
            logits = model(inputs)
            loss = F.cross_entropy(logits, targets)
        loss.backward()
    finally:
        for handle in handles:
            handle.remove()

    gradient_dtypes = sorted({str(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None})
    parameter_dtypes = sorted({str(parameter.dtype) for parameter in model.parameters()})

    return {
        "autocast_dtype": str(torch.bfloat16),
        "input_dtype": str(inputs.dtype),
        "parameter_dtypes_before_autocast": parameter_dtypes,
        "parameter_dtype_inside_autocast": parameter_dtype_inside_autocast,
        **activations,
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "gradient_dtypes": gradient_dtypes,
    }


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    accumulation = run_accumulation_experiment()
    toy_model = run_toy_model_experiment(device, args.seed)

    result = {
        "created_at_utc": datetime.now(UTC).isoformat(),
        "command": ["python", *sys.argv],
        "environment": {
            "python_version": sys.version.split()[0],
            "torch_version": torch.__version__,
            "cuda_runtime_version": torch.version.cuda,
            "gpu_name": torch.cuda.get_device_name(device),
        },
        "accumulation": accumulation,
        "toy_model": toy_model,
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print("mixed-precision accumulation:")
    for case in accumulation:
        print(f"  {case['name']}: result={case['result']:.10f} absolute_error={case['absolute_error_from_10']:.10f}")
    print("ToyModel dtypes under BF16 autocast:")
    for name, dtype in toy_model.items():
        print(f"  {name}: {dtype}")
    print(f"saved results: {args.output}")


if __name__ == "__main__":
    main()
