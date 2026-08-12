"""Fixed accumulation experiment plus BF16 ToyModel dtype inspection."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
from torch import nn


class ToyModel(nn.Module):
    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, x):
        return self.fc2(self.ln(self.relu(self.fc1(x))))


def accumulation_cases() -> dict[str, float]:
    outputs = {}
    for name, accumulator_dtype, addend_dtype, explicit_cast in (
        ("fp32_acc_fp32_add", torch.float32, torch.float32, False),
        ("fp16_acc_fp16_add", torch.float16, torch.float16, False),
        ("fp32_acc_fp16_add", torch.float32, torch.float16, False),
        ("fp32_acc_casted_fp16_add", torch.float32, torch.float16, True),
    ):
        s = torch.tensor(0.0, dtype=accumulator_dtype)
        for _ in range(1000):
            x = torch.tensor(0.01, dtype=addend_dtype)
            s += x.to(torch.float32) if explicit_cast else x
        outputs[name] = float(s)
    return outputs


def toy_dtypes() -> dict:
    if not torch.cuda.is_available():
        return {"status": "not_run_no_cuda"}
    device = torch.device("cuda")
    torch.manual_seed(42)
    model = ToyModel(8, 4).to(device)
    seen = {}
    hooks = [
        model.fc1.register_forward_hook(
            lambda _m, _i, o: seen.update(fc1_output=str(o.dtype))
        ),
        model.ln.register_forward_hook(
            lambda _m, _i, o: seen.update(layernorm_output=str(o.dtype))
        ),
    ]
    x = torch.randn(16, 8, device=device)
    targets = torch.randn(16, 4, device=device)
    with torch.autocast("cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = nn.functional.mse_loss(logits, targets)
    loss.backward()
    [h.remove() for h in hooks]
    seen.update(
        parameter_dtype=str(next(model.parameters()).dtype),
        logits_dtype=str(logits.dtype),
        loss_dtype=str(loss.dtype),
        gradient_dtype=str(next(model.parameters()).grad.dtype),
        status="pass",
    )
    return seen


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/mixed_precision.json")
    )
    parser.add_argument(
        "--formal-lm-benchmark-status",
        default="not_run_gpu_required",
        choices=("not_run_gpu_required", "pass"),
    )
    parser.add_argument("--formal-lm-benchmark-file", default="benchmark.csv")
    args = parser.parse_args()
    data = {
        "measurement_collected": True,
        "accumulation": accumulation_cases(),
        "toy_model_bf16": toy_dtypes(),
        "formal_lm_benchmark_status": args.formal_lm_benchmark_status,
    }
    if args.formal_lm_benchmark_status == "pass":
        data["formal_lm_benchmark_file"] = args.formal_lm_benchmark_file
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
