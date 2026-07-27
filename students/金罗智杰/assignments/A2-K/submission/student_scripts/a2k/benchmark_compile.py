"""Compare eager and torch.compile attention and Transformer execution."""

from __future__ import annotations

import argparse
import gc
import time
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import torch._functorch.config as functorch_config
import triton

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.attention_measurement import measure_attention_phase
from student_scripts.a2k.common import (
    ALLOCATOR_LIMIT_MIB,
    HARD_LIMIT_MIB,
    configure_cuda_environment,
    environment_metadata,
    peak_memory,
    write_csv,
    write_json,
)

ATTENTION_CONFIGS = ((512, 64), (2048, 128), (8192, 128))
PHASES = ("forward", "backward", "forward_backward")
MODEL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10_000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--warmup-ms", type=int, default=100)
    parser.add_argument("--rep-ms", type=int, default=300)
    return parser.parse_args()


def base_row(**overrides: Any) -> dict[str, Any]:
    row = {
        "scope": "",
        "implementation": "",
        "model_size": "",
        "sequence_length": "",
        "head_dim": "",
        "batch_size": 1,
        "dtype": "bfloat16",
        "is_causal": True,
        "phase": "",
        "warmup_ms": "",
        "rep_ms": "",
        "cold_start_ms": "",
        "latency_p20_ms": "",
        "latency_p50_ms": "",
        "latency_p80_ms": "",
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "within_24gib": "",
        "status": "",
        "error": "",
    }
    row.update(overrides)
    return row


def error_fields(error: Exception) -> dict[str, Any]:
    summary = " ".join(str(error).split())[:300]
    return {
        "status": "oom" if isinstance(error, torch.OutOfMemoryError) else "error",
        "error": f"{type(error).__name__}: {summary}",
        "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
        "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
        "within_24gib": torch.cuda.max_memory_reserved() / 1024**2 <= HARD_LIMIT_MIB,
    }


def attention_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for sequence_length, head_dim in ATTENTION_CONFIGS:
        for implementation in ("eager_pytorch", "compiled_pytorch"):
            def eager_function(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
                return explicit_attention(q, k, v, is_causal=True)

            function = torch.compile(eager_function, fullgraph=True) if implementation == "compiled_pytorch" else eager_function
            for phase in PHASES:
                row = base_row(
                    scope="attention",
                    implementation=implementation,
                    sequence_length=sequence_length,
                    head_dim=head_dim,
                    phase=phase,
                    warmup_ms=args.warmup_ms,
                    rep_ms=args.rep_ms,
                )
                q = k = v = None
                try:
                    requires_grad = phase != "forward"
                    q = torch.randn(
                        1,
                        sequence_length,
                        head_dim,
                        device="cuda",
                        dtype=torch.bfloat16,
                        requires_grad=requires_grad,
                    )
                    k = torch.randn_like(q, requires_grad=requires_grad)
                    v = torch.randn_like(q, requires_grad=requires_grad)
                    measurement = measure_attention_phase(
                        function,
                        q,
                        k,
                        v,
                        phase,
                        warmup_ms=args.warmup_ms,
                        rep_ms=args.rep_ms,
                    )
                    row.update(measurement)
                    row["within_24gib"] = float(measurement["peak_reserved_mib"]) <= HARD_LIMIT_MIB
                    row["status"] = "success"
                except Exception as error:
                    row.update(error_fields(error))
                rows.append(row)
                q = k = v = None
                gc.collect()
                torch.cuda.empty_cache()
                print(
                    f"attention {implementation=} {sequence_length=} {head_dim=} {phase=} "
                    f"status={row['status']}"
                )
    return rows


def measure_model_phase(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    targets: torch.Tensor,
    phase: str,
    warmup_ms: int,
    rep_ms: int,
) -> dict[str, float]:
    def forward_loss() -> tuple[torch.Tensor, torch.Tensor]:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(token_ids)
            loss = F.cross_entropy(logits.flatten(0, 1), targets.flatten())
        return logits, loss

    if phase == "forward":

        def operation() -> None:
            with torch.no_grad():
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(token_ids)

    elif phase == "forward_backward":

        def operation() -> None:
            optimizer.zero_grad(set_to_none=True)
            _, loss = forward_loss()
            loss.backward()

    elif phase == "train_step":

        def operation() -> None:
            optimizer.zero_grad(set_to_none=True)
            _, loss = forward_loss()
            loss.backward()
            optimizer.step()

    else:
        raise ValueError(f"unknown model phase: {phase}")

    torch.cuda.synchronize()
    cold_start = time.perf_counter()
    operation()
    torch.cuda.synchronize()
    cold_start_ms = (time.perf_counter() - cold_start) * 1000
    optimizer.zero_grad(set_to_none=True)
    torch.cuda.reset_peak_memory_stats()
    p20, p50, p80 = triton.testing.do_bench(
        operation,
        warmup=warmup_ms,
        rep=rep_ms,
        quantiles=[0.2, 0.5, 0.8],
    )
    allocated, reserved = peak_memory()
    return {
        "cold_start_ms": cold_start_ms,
        "latency_p20_ms": float(p20),
        "latency_p50_ms": float(p50),
        "latency_p80_ms": float(p80),
        "peak_allocated_mib": allocated,
        "peak_reserved_mib": reserved,
    }


def model_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for implementation in ("eager_pytorch", "compiled_pytorch"):
        torch.manual_seed(args.seed)
        model = BasicsTransformerLM(**MODEL_CONFIG).cuda()
        measured_model = torch.compile(model) if implementation == "compiled_pytorch" else model
        optimizer = torch.optim.AdamW(model.parameters(), lr=1.0e-3)
        token_ids = torch.randint(0, MODEL_CONFIG["vocab_size"], (1, MODEL_CONFIG["context_length"]), device="cuda")
        targets = torch.randint(0, MODEL_CONFIG["vocab_size"], (1, MODEL_CONFIG["context_length"]), device="cuda")
        for phase in ("forward", "forward_backward", "train_step"):
            row = base_row(
                scope="transformer",
                implementation=implementation,
                model_size="small",
                sequence_length=MODEL_CONFIG["context_length"],
                is_causal=True,
                phase=phase,
                warmup_ms=args.warmup_ms,
                rep_ms=args.rep_ms,
            )
            try:
                measurement = measure_model_phase(
                    measured_model,
                    optimizer,
                    token_ids,
                    targets,
                    phase,
                    args.warmup_ms,
                    args.rep_ms,
                )
                row.update(measurement)
                row["within_24gib"] = float(measurement["peak_reserved_mib"]) <= HARD_LIMIT_MIB
                row["status"] = "success"
            except Exception as error:
                row.update(error_fields(error))
            rows.append(row)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"transformer {implementation=} {phase=} status={row['status']}")
        model = measured_model = optimizer = token_ids = targets = None
        gc.collect()
        torch.cuda.empty_cache()
    return rows


def main() -> None:
    args = parse_args()
    # Backward-only timing reuses one graph with retain_graph=True. PyTorch's
    # donated-buffer mode explicitly forbids that measurement boundary.
    functorch_config.donated_buffer = False
    environment = configure_cuda_environment(require_rtx4090=True)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    rows = attention_rows(args) + model_rows(args)
    output_path = args.output_dir / "compile_comparison.csv"
    write_csv(output_path, rows)
    metadata = environment_metadata(
        environment,
        command="python student_scripts/a2k/benchmark_compile.py",
        seed=args.seed,
        warmup=f"{args.warmup_ms} ms",
        measurement=f"{args.rep_ms} ms",
    )
    metadata["compile_config"] = {"torch_functorch_donated_buffer": False}
    write_json(args.output_dir / "compile_comparison.metadata.json", metadata)
    print(f"saved {len(rows)} rows to {output_path}")


if __name__ == "__main__":
    main()
