from __future__ import annotations

import argparse
import math

import torch
from torch import nn

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k import explicit_attention
from student_scripts.a2k.common import add_common_args, ensure_dirs, peak_memory_mib, quantiles_ms, require_cuda, reset_peak, set_allocator_limit, write_csv


SMALL_CONFIG = {
    "vocab_size": 10000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10000.0,
}


def event_time(fn) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    fn()
    end.record()
    torch.cuda.synchronize()
    return float(start.elapsed_time(end))


def steady(fn) -> tuple[float, float, float]:
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    samples = [event_time(fn) for _ in range(20)]
    return quantiles_ms(samples)


def attention_row(sequence_length: int, head_dim: int, impl: str, device: torch.device) -> dict:
    q = torch.randn(1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, sequence_length, head_dim, device=device, dtype=torch.bfloat16, requires_grad=True)
    grad = torch.randn_like(q)
    fn = explicit_attention if impl == "eager" else torch.compile(explicit_attention)

    def step():
        q.grad = k.grad = v.grad = None
        out = fn(q, k, v, True)
        out.backward(grad)

    reset_peak()
    try:
        cold = "NA" if impl == "eager" else event_time(step)
        p20, p50, p80 = steady(step)
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cold = p20 = p50 = p80 = math.nan
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "OOM"
    return {
        "kind": "attention",
        "workload": "forward-backward",
        "sequence_length": sequence_length,
        "head_dim": head_dim,
        "implementation": impl,
        "cold_compile_ms": cold,
        "steady_p20_ms": p20,
        "steady_p50_ms": p50,
        "steady_p80_ms": p80,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "status": status,
    }


def model_row(workload: str, impl: str, device: torch.device) -> dict:
    model = BasicsTransformerLM(**SMALL_CONFIG).to(device)
    if impl == "compiled":
        model = torch.compile(model)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 512), device=device)
    targets = torch.randint(0, SMALL_CONFIG["vocab_size"], (1, 512), device=device)
    loss_fn = nn.CrossEntropyLoss()

    def forward():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(tokens)

    def forward_backward():
        optimizer.zero_grad(set_to_none=True)
        logits = forward()
        loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()

    def full_step():
        forward_backward()
        optimizer.step()

    fn = {"forward": forward, "forward-backward": forward_backward, "full training step": full_step}[workload]
    reset_peak()
    try:
        cold = "NA" if impl == "eager" else event_time(fn)
        p20, p50, p80 = steady(fn)
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "ok"
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        cold = p20 = p50 = p80 = math.nan
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "OOM"
    return {
        "kind": "small_model",
        "workload": workload,
        "sequence_length": 512,
        "head_dim": "NA",
        "implementation": impl,
        "cold_compile_ms": cold,
        "steady_p20_ms": p20,
        "steady_p50_ms": p50,
        "steady_p80_ms": p80,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    ensure_dirs()
    set_allocator_limit()
    device = require_cuda()
    torch.manual_seed(args.seed)

    rows = []
    for shape in [(512, 64), (2048, 128), (8192, 128)]:
        for impl in ["eager", "compiled"]:
            rows.append(attention_row(*shape, impl, device))
    for workload in ["forward", "forward-backward", "full training step"]:
        for impl in ["eager", "compiled"]:
            rows.append(model_row(workload, impl, device))

    write_csv(
        args.output_dir / "compile_comparison.csv",
        rows,
        [
            "kind",
            "workload",
            "sequence_length",
            "head_dim",
            "implementation",
            "cold_compile_ms",
            "steady_p20_ms",
            "steady_p50_ms",
            "steady_p80_ms",
            "peak_allocated_mib",
            "peak_reserved_mib",
            "status",
        ],
    )


if __name__ == "__main__":
    main()
