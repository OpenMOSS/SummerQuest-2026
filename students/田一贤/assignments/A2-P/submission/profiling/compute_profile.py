"""Framework-level CUDA profile with named model phases."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from .common import autocast_context, build_model, loss_from_logits, synchronize
from .config import get_model_spec


def _register_attention_ranges(model: torch.nn.Module) -> list[torch.utils.hooks.RemovableHandle]:
    """Annotate each attention-module forward without changing model numerics."""
    handles = []
    for module in model.modules():
        if "attention" not in type(module).__name__.lower():
            continue

        def before(attention_module, _inputs):
            context = record_function("attention")
            context.__enter__()
            attention_module._a2p_attention_context = context

        def after(attention_module, _inputs, output):
            context = attention_module.__dict__.pop("_a2p_attention_context", None)
            if context is not None:
                context.__exit__(None, None, None)
            return output

        handles.append(module.register_forward_pre_hook(before))
        handles.append(module.register_forward_hook(after))
    return handles


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", default="small")
    parser.add_argument("--context-length", type=int, default=512)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--dtype", choices=("fp32", "bf16"), default="fp32")
    parser.add_argument("--warmup-steps", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--trace-name", default="trace.json")
    parser.add_argument("--output-dir", type=Path, default=Path("results/profile"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if not torch.cuda.is_available():
        (args.output_dir / "run_metadata.json").write_text(
            json.dumps(
                {"status": "not_run_no_cuda", "measurement_collected": False}, indent=2
            ),
            encoding="utf-8",
        )
        (args.output_dir / "trace_summary.csv").write_text(
            "name,calls,cpu_time_us,cuda_time_us,status\n,,, ,not_run_no_cuda\n",
            encoding="utf-8",
        )
        (args.output_dir / "stage_summary.csv").write_text(
            "stage,calls,cpu_self_us,cpu_total_us,cuda_self_us,cuda_total_us,status\n"
            "forward,0,0,0,0,0,not_run_no_cuda\n"
            "attention,0,0,0,0,0,not_run_no_cuda\n"
            "backward,0,0,0,0,0,not_run_no_cuda\n"
            "optimizer,0,0,0,0,0,not_run_no_cuda\n",
            encoding="utf-8",
        )
        return
    device = torch.device("cuda")
    spec = get_model_spec(args.model_size)
    torch.manual_seed(args.seed)
    model = build_model(args.model_size, args.context_length, device)
    attention_handles = _register_attention_ranges(model)
    tokens = torch.randint(
        0, spec.vocab_size, (args.batch_size, args.context_length), device=device
    )
    targets = torch.randint_like(tokens, high=spec.vocab_size)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    optimizer.zero_grad(set_to_none=True)
    for _ in range(args.warmup_steps):
        optimizer.zero_grad(set_to_none=True)
        with autocast_context(device, args.dtype):
            warmup_logits = model(tokens)
            warmup_loss = loss_from_logits(warmup_logits, targets)
        warmup_loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    synchronize(device)
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=True,
    ) as prof:
        with record_function("train_step"):
            optimizer.zero_grad(set_to_none=True)
            with record_function("forward"):
                with autocast_context(device, args.dtype):
                    logits = model(tokens)
                    loss = loss_from_logits(logits, targets)
            with record_function("backward"):
                loss.backward()
            with record_function("optimizer"):
                optimizer.step()
    synchronize(device)
    for handle in attention_handles:
        handle.remove()
    trace_path = args.output_dir / args.trace_name
    prof.export_chrome_trace(str(trace_path))
    trace_sha256 = hashlib.sha256(trace_path.read_bytes()).hexdigest()
    averages = list(prof.key_averages())
    averages.sort(
        key=lambda event: float(getattr(event, "self_device_time_total", 0.0)),
        reverse=True,
    )
    rows = [
        {
            "name": event.key,
            "calls": event.count,
            "cpu_time_us": float(event.self_cpu_time_total),
            "cuda_time_us": float(
                getattr(event, "self_device_time_total", 0.0)
            ),
            "status": "measured",
        }
        for event in averages[:30]
    ]
    with (args.output_dir / "trace_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    by_name = {event.key: event for event in averages}
    stage_rows = []
    for stage in ("forward", "attention", "backward", "optimizer"):
        event = by_name.get(stage)
        stage_rows.append(
            {
                "stage": stage,
                "calls": event.count if event is not None else 0,
                "cpu_self_us": float(event.self_cpu_time_total)
                if event is not None
                else 0.0,
                "cpu_total_us": float(event.cpu_time_total)
                if event is not None
                else 0.0,
                "cuda_self_us": float(
                    getattr(event, "self_device_time_total", 0.0)
                )
                if event is not None
                else 0.0,
                "cuda_total_us": float(getattr(event, "device_time_total", 0.0))
                if event is not None
                else 0.0,
                "status": "measured" if event is not None else "missing",
            }
        )
    with (args.output_dir / "stage_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(f, fieldnames=list(stage_rows[0]))
        writer.writeheader()
        writer.writerows(stage_rows)
    metadata = {
        "status": "pass",
        "measurement_collected": True,
        "tool": "torch.profiler",
        "model_size": args.model_size,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "dtype": args.dtype,
        "seed": args.seed,
        "profile_range": "train_step",
        "stage_ranges": ["forward", "attention", "backward", "optimizer"],
        "profiled_steps": 1,
        "warmup_train_step_steps": args.warmup_steps,
        "trace_file": trace_path.name,
        "command": " ".join(__import__("sys").argv),
        "raw_trace": {
            "sha256": trace_sha256,
            "submitted": False,
            "retention": "remote execution workspace only",
        },
    }
    (args.output_dir / "run_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
