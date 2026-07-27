"""Profiling 章节中两道混合精度观察题的可运行实验。

``accumulation`` 对比 FP32/FP16 存储与累加的误差；``toy-dtypes`` 在 GPU
autocast 中给 ToyModel 各模块注册 hook，直接记录参数、激活、logits、loss 和
梯度 dtype。结果以 JSON 输出，避免只凭文档描述猜测当前 PyTorch 的实际策略。
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
from torch import nn


def run_accumulation() -> dict[str, Any]:
    """重复累加 1000 个 0.01，展示存储精度和累加精度的区别。"""

    cases = (
        ("fp32_value_fp32_accumulator", torch.float32, torch.float32, False),
        ("fp16_value_fp16_accumulator", torch.float16, torch.float16, False),
        ("fp16_value_fp32_inplace_accumulator", torch.float16, torch.float32, False),
        ("fp16_value_explicit_fp32_accumulator", torch.float16, torch.float32, True),
    )
    values: dict[str, dict[str, float | str]] = {}
    for name, value_dtype, accumulator_dtype, explicit_cast in cases:
        accumulator = torch.tensor(0.0, dtype=accumulator_dtype)
        for _ in range(1000):
            value = torch.tensor(0.01, dtype=value_dtype)
            accumulator += value.float() if explicit_cast else value
        measured = float(accumulator)
        values[name] = {
            "value_dtype": str(value_dtype),
            "accumulator_dtype": str(accumulator_dtype),
            "result": measured,
            "absolute_error_from_10": abs(measured - 10.0),
        }
    return {"event": "mixed_precision_accumulation", "status": "passed", "cases": values}


class ToyModel(nn.Module):
    """讲义给出的两层线性层 + LayerNorm 小模型。"""

    def __init__(self, in_features: int, out_features: int):
        super().__init__()
        self.fc1 = nn.Linear(in_features, 10, bias=False)
        self.ln = nn.LayerNorm(10)
        self.fc2 = nn.Linear(10, out_features, bias=False)
        self.relu = nn.ReLU()

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        hidden = self.relu(self.fc1(inputs))
        normalized = self.ln(hidden)
        return self.fc2(normalized)


def run_toy_dtypes(args: argparse.Namespace) -> dict[str, Any]:
    """记录 autocast 前后各组件真实 dtype。"""

    if not torch.cuda.is_available():
        raise RuntimeError("toy dtype experiment requires CUDA")
    model = ToyModel(args.in_features, args.out_features).cuda()
    inputs = torch.randn(args.batch_size, args.in_features, device="cuda")
    targets = torch.randn(args.batch_size, args.out_features, device="cuda")
    activations: dict[str, str] = {}
    handles = []
    for name, module in (("fc1", model.fc1), ("layer_norm", model.ln), ("fc2", model.fc2)):
        handles.append(
            module.register_forward_hook(
                lambda _module, _inputs, output, module_name=name: activations.__setitem__(module_name, str(output.dtype))
            )
        )

    autocast_dtype = {"fp16": torch.float16, "bf16": torch.bfloat16}[args.dtype]
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits = model(inputs)
        loss = torch.nn.functional.mse_loss(logits, targets)
    loss.backward()
    for handle in handles:
        handle.remove()
    return {
        "event": "mixed_precision_toy_dtypes",
        "status": "passed",
        "autocast_dtype": str(autocast_dtype),
        "parameter_dtypes": sorted({str(parameter.dtype) for parameter in model.parameters()}),
        "activation_dtypes": activations,
        "logits_dtype": str(logits.dtype),
        "loss_dtype": str(loss.dtype),
        "gradient_dtypes": sorted({str(parameter.grad.dtype) for parameter in model.parameters()}),
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run A2 mixed-precision observations.")
    subparsers = parser.add_subparsers(dest="experiment", required=True)
    subparsers.add_parser("accumulation")
    toy = subparsers.add_parser("toy-dtypes")
    toy.add_argument("--dtype", choices=("fp16", "bf16"), default="fp16")
    toy.add_argument("--in-features", type=int, default=128)
    toy.add_argument("--out-features", type=int, default=64)
    toy.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    result = run_accumulation() if args.experiment == "accumulation" else run_toy_dtypes(args)
    print(json.dumps(result, indent=2, ensure_ascii=False))
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("a", encoding="utf-8") as output:
            output.write(json.dumps(result, ensure_ascii=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
