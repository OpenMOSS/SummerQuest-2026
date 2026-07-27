from __future__ import annotations

import argparse
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from student_scripts.a2k.common import (
    append_csv,
    benchmark_quantiles,
    command_string,
    configure_formal_process,
    memory_peaks,
    public_environment,
    reset_peaks,
    write_json,
)

FIELDS = [
    "scope",
    "implementation",
    "model_size",
    "batch_size",
    "context_length",
    "dtype",
    "phase",
    "compile_cold_start_ms",
    "latency_ms_p20",
    "latency_ms_p50",
    "latency_ms_p80",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error_type",
    "command",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare eager and compiled small Transformer execution.")
    parser.add_argument("--implementation", choices=["eager", "compiled"], required=True)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = configure_formal_process()
    environment = public_environment(fraction)
    torch.manual_seed(args.seed)
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=512,
        d_model=768,
        d_ff=3072,
        num_layers=12,
        num_heads=12,
    ).cuda()
    if args.implementation == "compiled":
        model = torch.compile(model, fullgraph=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(0, 10_000, (1, 513), device="cuda")
    inputs, targets = tokens[:, :-1], tokens[:, 1:]

    def forward():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return model(inputs)

    def forward_backward():
        optimizer.zero_grad(set_to_none=True)
        logits = forward()
        loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()

    def train_step():
        forward_backward()
        optimizer.step()

    cold_start_ms = ""
    if args.implementation == "compiled":
        torch.cuda.synchronize()
        start = time.perf_counter()
        forward()
        torch.cuda.synchronize()
        cold_start_ms = (time.perf_counter() - start) * 1000

    rows = []
    for phase, function in (("forward", forward), ("forward_backward", forward_backward), ("train_step", train_step)):
        try:
            function()
            torch.cuda.synchronize()
            reset_peaks()
            p20, p50, p80 = benchmark_quantiles(function)
            peak_allocated, peak_reserved = memory_peaks()
            row = {
                "scope": "small_transformer",
                "implementation": args.implementation,
                "model_size": "small",
                "batch_size": 1,
                "context_length": 512,
                "dtype": "bf16_autocast_fp32_parameters",
                "phase": phase,
                "compile_cold_start_ms": cold_start_ms,
                "latency_ms_p20": p20,
                "latency_ms_p50": p50,
                "latency_ms_p80": p80,
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
                "status": "ok",
                "error_type": "",
                "command": command_string(),
            }
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            peak_allocated, peak_reserved = memory_peaks()
            row = {
                "scope": "small_transformer",
                "implementation": args.implementation,
                "model_size": "small",
                "batch_size": 1,
                "context_length": 512,
                "dtype": "bf16_autocast_fp32_parameters",
                "phase": phase,
                "compile_cold_start_ms": cold_start_ms,
                "latency_ms_p20": "",
                "latency_ms_p50": "",
                "latency_ms_p80": "",
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
                "status": "oom" if isinstance(exc, torch.cuda.OutOfMemoryError) else "error",
                "error_type": type(exc).__name__,
                "command": command_string(),
            }
        append_csv(args.output, row, FIELDS)
        rows.append(row)
    write_json(args.metadata, {"environment": environment, "seed": args.seed, "latest_rows": rows})
    print([(row["phase"], row["status"], row["latency_ms_p50"]) for row in rows])


if __name__ == "__main__":
    main()
