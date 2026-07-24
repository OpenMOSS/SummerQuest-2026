from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from profiling.common import (
    add_common_args,
    build_model,
    collect_metadata,
    cuda_sync,
    lm_loss,
    make_batch,
    resolve_device,
    set_seed,
    write_json,
)


def install_attention_ranges() -> None:
    import cs336_basics.model as model_mod

    def profiled_attention(Q: torch.Tensor, K: torch.Tensor, V: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        with record_function("attention/scores"):
            scores = torch.einsum("...qd,...kd->...qk", Q, K) / math.sqrt(K.shape[-1])
            if mask is not None:
                scores = torch.where(mask, scores, float("-inf"))
        with record_function("attention/softmax"):
            weights = torch.softmax(scores, dim=-1)
        with record_function("attention/value"):
            return torch.einsum("...qk,...kd->...qd", weights, V)

    model_mod.scaled_dot_product_attention = profiled_attention


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one stable train_step trace with torch.profiler.")
    add_common_args(parser)
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--trace-output", type=Path, required=True)
    return parser.parse_args()


def train_step_with_ranges(model: torch.nn.Module, optimizer: torch.optim.Optimizer, x: torch.Tensor, y: torch.Tensor, dtype: str, device: torch.device) -> torch.Tensor:
    optimizer.zero_grad(set_to_none=True)
    with record_function("forward"):
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=dtype == "bf16" and device.type == "cuda"):
            logits = model(x)
            loss = lm_loss(logits, y)
    with record_function("backward"):
        loss.backward()
    with record_function("optimizer"):
        optimizer.step()
    return loss.detach()


def summarize_events(prof: profile) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for event in prof.key_averages():
        device_total = getattr(event, "device_time_total", 0.0) or getattr(event, "cuda_time_total", 0.0)
        self_device_total = getattr(event, "self_device_time_total", 0.0) or getattr(event, "self_cuda_time_total", 0.0)
        rows.append(
            {
                "name": event.key,
                "calls": event.count,
                "cpu_time_total_us": event.cpu_time_total,
                "cuda_time_total_us": device_total,
                "self_cpu_time_total_us": event.self_cpu_time_total,
                "self_cuda_time_total_us": self_device_total,
            }
        )
    rows.sort(key=lambda row: float(row["cuda_time_total_us"] or row["cpu_time_total_us"]), reverse=True)
    return rows


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    set_seed(args.seed)
    install_attention_ranges()
    model = build_model(args.model_size, args.context_length, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    x, y = make_batch(args.model_size, args.batch_size, args.context_length, device)

    for _ in range(args.warmup):
        with record_function("profile/warmup"):
            train_step_with_ranges(model, optimizer, x, y, args.dtype, device)
        cuda_sync(device)

    activities = [ProfilerActivity.CPU]
    if device.type == "cuda":
        activities.append(ProfilerActivity.CUDA)

    with profile(activities=activities, record_shapes=True, profile_memory=True, with_stack=False) as prof:
        with record_function("profile/measure"):
            loss = train_step_with_ranges(model, optimizer, x, y, args.dtype, device)
            cuda_sync(device)

    args.trace_output.parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(args.trace_output))
    rows = summarize_events(prof)
    write_json(
        args.output,
        {
            "config": {
                "model_size": args.model_size,
                "batch_size": args.batch_size,
                "context_length": args.context_length,
                "dtype": args.dtype,
                "warmup": args.warmup,
                "seed": args.seed,
                "tool": "torch.profiler",
                "trace_file": args.trace_output.name,
                "loss": float(loss.item()),
            },
            "events": rows,
        },
    )
    write_json(args.metadata_output or args.output.with_name("run_metadata.json"), collect_metadata(args, {"tool": "torch.profiler"}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
