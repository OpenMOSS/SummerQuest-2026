from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch
import torch.nn as nn
import torch.nn.functional as F

from profiling.benchmark import configure_runtime
from profiling.config import (
    DEFAULT_SEED,
    base_metadata,
    classify_error,
    environment_metadata,
    public_relative_path,
    safe_error_summary,
    utc_now,
    write_json,
)


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
        return self.fc2(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-P mixed-precision experiments")
    subparsers = parser.add_subparsers(dest="experiment", required=True)

    accumulation = subparsers.add_parser("accumulation")
    accumulation.add_argument("--run-id", default="MP-A")
    accumulation.add_argument("--output", type=Path, required=True)

    toy = subparsers.add_parser("toy")
    toy.add_argument("--run-id", default="MP-T")
    toy.add_argument("--output", type=Path, required=True)
    toy.add_argument("--device", default="cuda")
    toy.add_argument("--dtype", choices=("fp32", "bf16"), default="bf16")
    toy.add_argument("--seed", type=int, default=DEFAULT_SEED)
    toy.add_argument("--batch-size", type=int, default=32)
    toy.add_argument("--in-features", type=int, default=128)
    toy.add_argument("--out-features", type=int, default=32)
    return parser.parse_args()


def _accumulation_result(value: torch.Tensor) -> float:
    return float(value.item())


def _run_fixed_accumulation_cases() -> list[tuple[str, torch.dtype, torch.dtype, bool, float]]:
    # Keep these four blocks structurally identical to the fixed assignment PDF.
    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    case_1 = _accumulation_result(s)

    s = torch.tensor(0, dtype=torch.float16)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    case_2 = _accumulation_result(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    case_3 = _accumulation_result(s)

    s = torch.tensor(0, dtype=torch.float32)
    for _ in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    case_4 = _accumulation_result(s)

    return [
        ("fp32_input_fp32_accumulator", torch.float32, torch.float32, False, case_1),
        ("fp16_input_fp16_accumulator", torch.float16, torch.float16, False, case_2),
        ("fp16_input_fp32_accumulator_implicit", torch.float32, torch.float16, False, case_3),
        ("fp16_input_fp32_accumulator_explicit", torch.float32, torch.float16, True, case_4),
    ]


def run_accumulation(args: argparse.Namespace) -> dict[str, Any]:
    reference = 10.0
    results = []
    for name, accumulator_dtype, input_dtype, explicit_cast, value in _run_fixed_accumulation_cases():
        absolute_error = abs(value - reference)
        results.append(
            {
                "name": name,
                "accumulator_dtype": str(accumulator_dtype),
                "input_dtype": str(input_dtype),
                "explicit_cast_to_accumulator": explicit_cast,
                "value": value,
                "reference": reference,
                "absolute_error": absolute_error,
                "relative_error": absolute_error / reference,
            }
        )

    payload = base_metadata(args.run_id, "mixed_precision_accumulation", "torch")
    payload.update(
        {
            "kind": "mixed_precision_accumulation",
            "status": "success",
            "results": results,
            "environment": environment_metadata(torch),
            "output_file": public_relative_path(args.output),
            "finished_at": utc_now(),
        }
    )
    write_json(args.output, payload)
    return payload


def _dtype_name(value: torch.dtype) -> str:
    return str(value).removeprefix("torch.")


def run_toy(args: argparse.Namespace) -> dict[str, Any]:
    payload = base_metadata(args.run_id, f"toy_model_{args.dtype}", "torch.autocast")
    payload.update(
        {
            "kind": "mixed_precision_toy_model",
            "dtype": args.dtype,
            "seed": args.seed,
            "batch_size": args.batch_size,
            "in_features": args.in_features,
            "out_features": args.out_features,
            "output_file": public_relative_path(args.output),
        }
    )
    failure_stage = "initialization"
    try:
        configure_runtime(args.seed)
        device = torch.device(args.device)
        if device.type != "cuda":
            raise ValueError("ToyModel dtype experiment requires CUDA")
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false")
        if args.dtype == "bf16" and not torch.cuda.is_bf16_supported():
            raise RuntimeError("the selected CUDA device does not report BF16 support")

        model = ToyModel(args.in_features, args.out_features).to(device)
        inputs = torch.randn(args.batch_size, args.in_features, device=device)
        targets = torch.randint(0, args.out_features, (args.batch_size,), device=device)
        observed: dict[str, str] = {}

        def capture(name: str):
            def hook(_module: nn.Module, _inputs: tuple[torch.Tensor, ...], output: torch.Tensor) -> None:
                observed[name] = _dtype_name(output.dtype)

            return hook

        handles = [
            model.fc1.register_forward_hook(capture("first_layer_output")),
            model.ln.register_forward_hook(capture("layer_norm_output")),
        ]
        try:
            failure_stage = "forward"
            autocast = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if args.dtype == "bf16" else torch.autocast(device_type="cuda", enabled=False)
            with autocast:
                logits = model(inputs)
                loss = F.cross_entropy(logits, targets)
            observed["logits"] = _dtype_name(logits.dtype)
            observed["loss"] = _dtype_name(loss.dtype)

            failure_stage = "backward"
            loss.backward()
            gradient_dtypes = sorted({_dtype_name(parameter.grad.dtype) for parameter in model.parameters() if parameter.grad is not None})
            parameter_dtypes = sorted({_dtype_name(parameter.dtype) for parameter in model.parameters()})
            torch.cuda.synchronize(device)

            payload.update(
                {
                    "status": "success",
                    "parameter_dtypes": parameter_dtypes,
                    "observed_dtypes": observed,
                    "gradient_dtypes": gradient_dtypes,
                    "loss_value": float(loss.detach().float().item()),
                    "logits_finite": bool(torch.isfinite(logits).all().item()),
                    "environment": environment_metadata(torch),
                }
            )
        finally:
            for handle in handles:
                handle.remove()
    except Exception as exc:
        payload.update(
            {
                "status": classify_error(exc),
                "failure_stage": failure_stage,
                "error_type": exc.__class__.__name__,
                "error_summary": safe_error_summary(exc),
                "environment": environment_metadata(torch),
            }
        )
        raise
    finally:
        payload["finished_at"] = utc_now()
        write_json(args.output, payload)
    return payload


def main() -> int:
    args = parse_args()
    if args.experiment == "accumulation":
        run_accumulation(args)
    else:
        run_toy(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
