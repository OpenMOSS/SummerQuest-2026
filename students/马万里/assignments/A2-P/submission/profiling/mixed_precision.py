import argparse
import json
import statistics
import time
from pathlib import Path

import torch
import torch.nn as nn

from env_utils import public_env

RESULT_PATH = Path("results/mixed_precision.json")


# ---------- 1. Mixed-precision accumulation ----------
def run_accumulation() -> dict:
    def vary(acc_dtype, add_dtype, cast=False):
        s = torch.tensor(0, dtype=acc_dtype)
        for _ in range(1000):
            a = torch.tensor(0.01, dtype=add_dtype)
            a = a.type(torch.float32) if cast else a
            s += a
        return s.item()

    out = {
        "reference": 10.0,
        "fp32_acc_fp32_add": vary(torch.float32, torch.float32),
        "fp16_acc_fp16_add": vary(torch.float16, torch.float16),
        "fp32_acc_fp16_add": vary(torch.float32, torch.float16),
        "fp32_acc_cast_fp16_to_fp32_add": vary(torch.float32, torch.float16, cast=True),
    }
    out["fp16_acc_error"] = out["fp16_acc_fp16_add"] - out["reference"]
    out["fp16_addend_error"] = out["fp32_acc_fp16_add"] - out["reference"]
    return out


# ---------- 2. ToyModel dtype under autocast ----------
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


def record_dtypes(dtype, in_features=64, out_features=10) -> dict:
    torch.manual_seed(0)
    model = ToyModel(in_features, out_features).cuda()
    x = torch.randn(4, in_features, device="cuda")
    with torch.autocast(device_type="cuda", dtype=dtype):
        h1 = model.fc1(x)
        relu = model.relu(h1)
        ln = model.ln(relu)
        logits = model.fc2(ln)
        loss = nn.functional.cross_entropy(
            logits, torch.randint(0, out_features, (4,), device="cuda"))
    loss.backward()
    return {
        "autocast_dtype": str(dtype),
        "param_fc1": str(model.fc1.weight.dtype),
        "fc1_output": str(h1.dtype),
        "relu_output": str(relu.dtype),
        "layernorm_output": str(ln.dtype),
        "logits": str(logits.dtype),
        "loss": str(loss.dtype),
        "fc1_grad": str(model.fc1.weight.grad.dtype),
        "layernorm_grad": str(model.ln.weight.grad.dtype),
        "fc2_grad": str(model.fc2.weight.grad.dtype),
    }


# ---------- 3. FP32 vs BF16-autocast benchmark ----------
def toy_bench(mode, in_features=64, out_features=10, batch=32, warmup=5, steps=10) -> dict:
    torch.manual_seed(0)
    model = ToyModel(in_features, out_features).cuda()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(batch, in_features, device="cuda")
    tgt = torch.randint(0, out_features, (batch,), device="cuda")

    def step():
        model.zero_grad(set_to_none=True)
        if mode == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, tgt)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, tgt)
        loss.backward()
        opt.step()

    for _ in range(warmup):
        step()
        torch.cuda.synchronize()

    torch.cuda.reset_peak_memory_stats()
    times = []
    for _ in range(steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        step()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)
    peak_mib = torch.cuda.max_memory_allocated() / 1024 ** 2

    # 数值趋势：一次 forward 的 loss 与 logits 量级
    with torch.no_grad():
        if mode == "bf16":
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(x)
                loss = nn.functional.cross_entropy(logits, tgt)
        else:
            logits = model(x)
            loss = nn.functional.cross_entropy(logits, tgt)

    return {
        "mode": mode,
        "batch_size": batch,
        "mean_ms": statistics.mean(times) * 1e3,
        "std_ms": statistics.stdev(times) * 1e3,
        "peak_memory_mib": peak_mib,
        "loss": float(loss.item()),
        "logits_mean_abs": float(logits.abs().mean().item()),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", choices=["accumulation", "toy", "all"], default="all")
    ap.add_argument("--in-features", type=int, default=64)
    ap.add_argument("--out-features", type=int, default=10)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--steps", type=int, default=10)
    args = ap.parse_args()

    out = {"env": public_env()}
    if args.run in ("accumulation", "all"):
        out["accumulation"] = run_accumulation()
    if args.run in ("toy", "all"):
        if not torch.cuda.is_available():
            raise RuntimeError("toy experiments require CUDA; run on a GPU node.")
        out["toy_dtypes"] = {
            "fp16": record_dtypes(torch.float16, args.in_features, args.out_features),
            "bf16": record_dtypes(torch.bfloat16, args.in_features, args.out_features),
        }
        out["toy_bench"] = {
            "fp32": toy_bench("fp32", args.in_features, args.out_features,
                              args.batch, args.warmup, args.steps),
            "bf16_autocast": toy_bench("bf16", args.in_features, args.out_features,
                                       args.batch, args.warmup, args.steps),
        }

    RESULT_PATH.parent.mkdir(parents=True, exist_ok=True)
    RESULT_PATH.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(json.dumps(out, indent=2, default=str))
    print(f"wrote {RESULT_PATH}")


if __name__ == "__main__":
    main()
