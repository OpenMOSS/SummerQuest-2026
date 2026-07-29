"""A2-P task 3: mixed precision experiments.

(a) The four accumulation variants from the PDF, run verbatim.
(b) ToyModel under CUDA BF16 autocast: record dtypes of parameters, fc1
    output, layer-norm output, logits, loss, gradients.
Outputs a JSON consumed by summarize.py.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch
from torch import nn

sys.path.insert(0, str(Path(__file__).parent))
from common import save_json  # noqa: E402


def run_accumulation():
    results = {}
    # variant 1: fp32 accumulator + fp32 addend
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float32)
    results["fp32_accum_fp32_addend"] = float(s)

    # variant 2: fp16 accumulator + fp16 addend
    s = torch.tensor(0, dtype=torch.float16)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp16_accum_fp16_addend"] = float(s)

    # variant 3: fp32 accumulator + fp16 addend
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        s += torch.tensor(0.01, dtype=torch.float16)
    results["fp32_accum_fp16_addend"] = float(s)

    # variant 4: fp32 accumulator + fp16 addend upcast to fp32 before adding
    s = torch.tensor(0, dtype=torch.float32)
    for i in range(1000):
        x = torch.tensor(0.01, dtype=torch.float16)
        s += x.type(torch.float32)
    results["fp32_accum_fp16_addend_upcast"] = float(s)
    return results


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
        x = self.fc2(x)
        return x


def run_toymodel_bf16(device="cuda"):
    torch.manual_seed(0)
    model = ToyModel(32, 8).to(device)
    x = torch.randn(4, 32, device=device)
    captured = {}

    def hook(name):
        def fn(mod, inp, out):
            captured[name] = str(out.dtype)
        return fn

    model.fc1.register_forward_hook(hook("fc1_output"))
    model.ln.register_forward_hook(hook("ln_output"))

    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        logits = model(x)
        loss = logits.sum()
    loss.backward()
    return {
        "autocast_dtype": "bfloat16",
        "param_dtype_inside_autocast": str(model.fc1.weight.dtype),
        "fc1_output": captured["fc1_output"],
        "ln_output": captured["ln_output"],
        "logits": str(logits.dtype),
        "loss": str(loss.dtype),
        "gradient_fc1_weight": str(model.fc1.weight.grad.dtype),
        "gradient_ln_weight": str(model.ln.weight.grad.dtype),
    }


if __name__ == "__main__":
    out = sys.argv[1] if len(sys.argv) > 1 else "results/mixed_precision_raw.json"
    result = {
        "accumulation": run_accumulation(),
        "toymodel_bf16_autocast": run_toymodel_bf16(),
    }
    save_json(result, out)
    print(json.dumps(result, indent=2))
