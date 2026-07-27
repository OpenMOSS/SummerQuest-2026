from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.common import collect_metadata, cuda_sync, memory_stats, reset_peak_memory, resolve_device, set_seed, summarize_samples, write_json


class ToyModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc1 = torch.nn.Linear(10, 10, bias=False)
        self.ln = torch.nn.LayerNorm(10)
        self.fc2 = torch.nn.Linear(10, 10, bias=False)

    def forward(self, x: torch.Tensor, capture: dict[str, str] | None = None) -> torch.Tensor:
        x = self.fc1(x)
        if capture is not None:
            capture["first_layer_output"] = str(x.dtype)
        x = self.ln(x)
        if capture is not None:
            capture["layernorm_output"] = str(x.dtype)
        x = self.fc2(torch.relu(x))
        if capture is not None:
            capture["logits"] = str(x.dtype)
        return x


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Mixed precision experiments for A2-P.")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


def accumulation_experiments(device: torch.device) -> list[dict[str, object]]:
    n = 10000
    values_fp32 = torch.full((n,), 0.1, device=device, dtype=torch.float32)
    values_fp16 = values_fp32.to(torch.float16)
    return [
        {"case": "fp32_input_fp32_sum", "value": float(values_fp32.sum(dtype=torch.float32).item())},
        {"case": "fp16_input_fp16_sum", "value": float(values_fp16.sum(dtype=torch.float16).item())},
        {"case": "fp16_input_fp32_sum", "value": float(values_fp16.sum(dtype=torch.float32).item())},
        {"case": "fp32_input_then_fp16_sum", "value": float(values_fp32.to(torch.float16).sum(dtype=torch.float16).item())},
    ]


def run_toy(dtype: str, args: argparse.Namespace, device: torch.device) -> dict[str, object]:
    set_seed(args.seed)
    model = ToyModel().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randn(args.batch_size, 10, device=device)
    y = torch.randint(0, 10, (args.batch_size,), device=device)
    dtype_capture: dict[str, str] = {"parameters": str(next(model.parameters()).dtype)}

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype == "bf16" and device.type == "cuda"):
            logits = model(x)
            loss = F.cross_entropy(logits.float(), y)
        loss.backward()
        optimizer.step()
        cuda_sync(device)

    reset_peak_memory(device)
    samples: list[float] = []
    losses: list[float] = []
    for step in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        start = time.perf_counter()
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype == "bf16" and device.type == "cuda"):
            logits = model(x, dtype_capture if step == 0 else None)
            loss = F.cross_entropy(logits.float(), y)
            dtype_capture["loss"] = str(loss.dtype)
        loss.backward()
        dtype_capture["gradient"] = str(next(model.parameters()).grad.dtype)
        optimizer.step()
        cuda_sync(device)
        samples.append((time.perf_counter() - start) * 1000.0)
        losses.append(float(loss.item()))

    return {
        "dtype": dtype,
        "timings_ms": samples,
        "summary": summarize_samples(samples),
        "loss_first": losses[0],
        "loss_last": losses[-1],
        "dtypes": dtype_capture,
        "memory": memory_stats(device),
    }


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    payload = {
        "metadata": collect_metadata(args),
        "accumulation": accumulation_experiments(device),
        "toy_model": [run_toy("fp32", args, device), run_toy("bf16", args, device)],
    }
    write_json(Path(args.output), payload)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
