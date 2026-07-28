from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch
import triton

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k import FlashAttentionTriton, explicit_attention
from student_scripts.a2k.common import cuda_event_bench, peak_memory, require_cuda, reset_peak_memory, timing_summary, write_csv


def make_inputs(seq: int, d: int, dtype: torch.dtype, device: torch.device):
    torch.manual_seed(0)
    q = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=True)
    k = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=True)
    v = torch.randn(1, seq, d, device=device, dtype=dtype, requires_grad=True)
    do = torch.randn(1, seq, d, device=device, dtype=dtype)
    return q, k, v, do


def clear_grads(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> None:
    q.grad = None
    k.grad = None
    v.grad = None


def run_phase(fn, q, k, v, do, phase: str):
    def forward():
        out = fn(q, k, v)
        if phase == "forward":
            del out

    def backward():
        clear_grads(q, k, v)
        out = fn(q, k, v)
        out.backward(do, retain_graph=False)

    def forward_backward():
        clear_grads(q, k, v)
        out = fn(q, k, v)
        out.backward(do, retain_graph=False)

    target = forward
    if phase == "backward":
        out = fn(q, k, v)
        torch.cuda.synchronize()

        def target():
            clear_grads(q, k, v)
            out.backward(do, retain_graph=True)

    if phase == "forward_backward":
        target = forward_backward
    reset_peak_memory()
    ms = triton.testing.do_bench(target, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8], return_mode="all")
    mem = peak_memory()
    summary = timing_summary([float(x) for x in ms])
    return summary | mem


def measure_compile_cold(fn, q, k, v) -> float:
    torch.cuda.synchronize()
    start = time.perf_counter()
    out = fn(q, k, v)
    del out
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000


def run_small_model_comparison(device: torch.device) -> list[dict]:
    dtype = torch.bfloat16
    rows = []
    config = {
        "vocab_size": 10_000,
        "context_length": 512,
        "d_model": 768,
        "d_ff": 3072,
        "num_layers": 12,
        "num_heads": 12,
        "rope_theta": 10_000.0,
    }
    torch.manual_seed(11)
    tokens = torch.randint(0, config["vocab_size"], (1, config["context_length"]), device=device)
    targets = torch.randint(0, config["vocab_size"], (1, config["context_length"]), device=device)
    for implementation in ["eager", "compiled"]:
        torch.manual_seed(10)
        model = BasicsTransformerLM(**config).to(device)
        optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
        if implementation == "compiled":
            start = time.perf_counter()
            model = torch.compile(model, fullgraph=False)
            with torch.autocast(device_type="cuda", dtype=dtype):
                _ = model(tokens)
            torch.cuda.synchronize()
            cold_compile_ms = (time.perf_counter() - start) * 1000
        else:
            cold_compile_ms = ""

        def forward():
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens)
            del logits

        def forward_backward():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens)
                loss = cross_entropy(logits, targets)
            loss.backward()

        def train_step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=dtype):
                logits = model(tokens)
                loss = cross_entropy(logits, targets)
            loss.backward()
            optimizer.step()

        for phase, fn in [("model_forward", forward), ("model_forward_backward", forward_backward), ("model_train_step", train_step)]:
            try:
                reset_peak_memory()
                samples = cuda_event_bench(fn, warmup_steps=3, measurement_steps=5)
                mem = peak_memory()
                rows.append(
                    {
                        "kind": "small_transformer",
                        "implementation": implementation,
                        "sequence_length": 512,
                        "head_dim": 64,
                        "batch_size": 1,
                        "dtype": "bf16_autocast_fp32_params",
                        "causal": True,
                        "phase": phase,
                        "cold_compile_ms": cold_compile_ms if phase == "model_forward" else "",
                        "samples_ms": samples,
                        "p20_ms": "",
                        "p50_ms": sorted(samples)[len(samples) // 2],
                        "p80_ms": "",
                        "mean_ms": sum(samples) / len(samples),
                        "peak_allocated_mib": mem["peak_allocated_mib"],
                        "peak_reserved_mib": mem["peak_reserved_mib"],
                        "speedup_vs_eager_p50": "",
                        "q_tile_size": "",
                        "k_tile_size": "",
                        "num_warps": "",
                        "num_stages": "",
                        "status": "ok",
                    }
                )
            except torch.OutOfMemoryError as exc:
                torch.cuda.empty_cache()
                rows.append(
                    {
                        "kind": "small_transformer",
                        "implementation": implementation,
                        "sequence_length": 512,
                        "head_dim": 64,
                        "batch_size": 1,
                        "dtype": "bf16_autocast_fp32_params",
                        "causal": True,
                        "phase": phase,
                        "cold_compile_ms": cold_compile_ms if phase == "model_forward" else "",
                        "samples_ms": [],
                        "p20_ms": "",
                        "p50_ms": "",
                        "p80_ms": "",
                        "mean_ms": "",
                        "peak_allocated_mib": "",
                        "peak_reserved_mib": "",
                        "speedup_vs_eager_p50": "",
                        "q_tile_size": "",
                        "k_tile_size": "",
                        "num_warps": "",
                        "num_stages": "",
                        "status": f"oom: {type(exc).__name__}",
                    }
                )
        del model, optimizer
        torch.cuda.empty_cache()
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, default=Path("local_results/a2k"))
    parser.add_argument("--include-compile", action="store_true")
    parser.add_argument("--skip-small-model", action="store_true")
    args = parser.parse_args()
    device = require_cuda()
    dtype = torch.bfloat16
    rows = []
    for seq in [512, 2048, 8192, 16384]:
        for d in [64, 128]:
            for implementation in ["eager", "compiled", "triton"]:
                if seq == 16384 and implementation == "compiled" and not args.include_compile:
                    continue
                q, k, v, do = make_inputs(seq, d, dtype, device)
                if implementation == "eager":
                    fn = lambda q, k, v: explicit_attention(q, k, v, True)[0]
                    cold_compile_ms = ""
                elif implementation == "compiled":
                    fn = torch.compile(lambda q, k, v: explicit_attention(q, k, v, True)[0], fullgraph=True)
                    cold_compile_ms = measure_compile_cold(fn, q, k, v)
                else:
                    fn = lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True)
                    cold_compile_ms = ""
                for phase in ["forward", "backward", "forward_backward"]:
                    base = {
                        "kind": "attention",
                        "implementation": implementation,
                        "sequence_length": seq,
                        "head_dim": d,
                        "batch_size": 1,
                        "dtype": "bfloat16",
                        "causal": True,
                        "phase": phase,
                        "cold_compile_ms": cold_compile_ms if phase == "forward" else "",
                        "q_tile_size": 64 if implementation == "triton" else "",
                        "k_tile_size": 64 if implementation == "triton" else "",
                        "num_warps": 4 if implementation == "triton" and d <= 64 else (8 if implementation == "triton" else ""),
                        "num_stages": 3 if implementation == "triton" else "",
                    }
                    try:
                        result = run_phase(fn, q, k, v, do, phase)
                        rows.append(base | result | {"speedup_vs_eager_p50": "", "status": "ok"})
                    except torch.OutOfMemoryError as exc:
                        torch.cuda.empty_cache()
                        rows.append(base | {"samples_ms": [], "p20_ms": "", "p50_ms": "", "p80_ms": "", "mean_ms": "", "peak_allocated_mib": "", "peak_reserved_mib": "", "speedup_vs_eager_p50": "", "status": f"oom: {type(exc).__name__}"})
                    except Exception as exc:
                        torch.cuda.empty_cache()
                        rows.append(base | {"samples_ms": [], "p20_ms": "", "p50_ms": "", "p80_ms": "", "mean_ms": "", "peak_allocated_mib": "", "peak_reserved_mib": "", "speedup_vs_eager_p50": "", "status": f"error: {type(exc).__name__}: {exc}"})

    eager = {(r["sequence_length"], r["head_dim"], r["phase"]): r for r in rows if r["implementation"] == "eager" and r["status"] == "ok"}
    for row in rows:
        ref = eager.get((row["sequence_length"], row["head_dim"], row["phase"]))
        if row["status"] == "ok" and ref and row["implementation"] != "eager":
            row["speedup_vs_eager_p50"] = float(ref["p50_ms"]) / float(row["p50_ms"])

    write_csv(args.output_dir / "flash_benchmark.csv", rows)
    baseline = [r for r in rows if r["implementation"] == "eager" and r["sequence_length"] in [512, 2048, 8192]]
    write_csv(args.output_dir / "attention_baseline.csv", baseline)
    compile_shapes = {(512, 64), (2048, 128), (8192, 128)}
    compile_rows = [
        r
        for r in rows
        if r["implementation"] in ["eager", "compiled"]
        and (r["sequence_length"], r["head_dim"]) in compile_shapes
    ]
    if not args.skip_small_model:
        compile_rows.extend(run_small_model_comparison(device))
    write_csv(args.output_dir / "compile_comparison.csv", compile_rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
