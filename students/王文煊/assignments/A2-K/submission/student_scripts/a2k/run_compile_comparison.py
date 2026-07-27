"""A2-K task 2 (section 6.2): torch.compile comparison.

Part A: explicit attention at (512,64), (2048,128), (8192,128), eager vs
compiled (cold-start compile time recorded separately from steady-state).
Part B: Stanford small model, batch 1, ctx 512, BF16 autocast: eager vs
compiled forward / fwd+bwd / full training step.

Run:
    PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python student_scripts/a2k/run_compile_comparison.py
"""

from __future__ import annotations

import csv
import os
import sys
import time

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import (  # noqa: E402
    MODEL_SIZES,
    VOCAB_SIZE,
    collect_metadata,
    git_commit,
    mib,
    set_allocator_limit,
    write_json,
)

ALLOCATOR_FRACTION = set_allocator_limit()  # before any CUDA allocation

from cs336_basics.model import BasicsTransformerLM  # noqa: E402
from cs336_basics.optimizer import AdamW  # noqa: E402
from cs336_systems.a2k.attention import explicit_attention  # noqa: E402

OUT_DIR = os.path.join("local_results", "a2k")
SEED = 0
ATTN_CONFIGS = [(512, 64), (2048, 128), (8192, 128)]


def bench_attention(seq, d, impl):
    torch.manual_seed(SEED)
    q = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16)

    def fwd_bwd(fn):
        o = fn(q, k, v, True)
        o.backward(do)
        for t in (q, k, v):
            t.grad = None

    rows = []
    cold_s = None
    try:
        if impl == "compiled":
            fn = torch.compile(explicit_attention)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            fwd_bwd(fn)  # first call triggers compilation
            torch.cuda.synchronize()
            cold_s = round(time.perf_counter() - t0, 3)
        else:
            fn = explicit_attention
        q20, q50, q80 = triton.testing.do_bench(
            lambda: fwd_bwd(fn), warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
        )
        torch.cuda.reset_peak_memory_stats()
        fwd_bwd(fn)
        torch.cuda.synchronize()
        rows.append(
            {
                "experiment": "attention_fwd_bwd",
                "seq_len": seq,
                "head_dim": d,
                "implementation": impl,
                "phase": "fwd_bwd",
                "cold_start_s": cold_s,
                "p20_ms": q20,
                "p50_ms": q50,
                "p80_ms": q80,
                "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
                "status": "ok",
            }
        )
    except torch.cuda.OutOfMemoryError:
        rows.append(
            {
                "experiment": "attention_fwd_bwd",
                "seq_len": seq,
                "head_dim": d,
                "implementation": impl,
                "phase": "fwd_bwd",
                "cold_start_s": cold_s,
                "p20_ms": None,
                "p50_ms": None,
                "p80_ms": None,
                "peak_allocated_mib": "",
                "peak_reserved_mib": "",
                "status": "oom",
            }
        )
    del q, k, v, do
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return rows


def make_small_model():
    d_model, d_ff, num_layers, num_heads = MODEL_SIZES["small"]
    torch.manual_seed(SEED)
    return BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=512,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    ).cuda()


def bench_model(impl):
    ctx = 512
    model = make_small_model()
    opt = AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randint(0, VOCAB_SIZE, (1, ctx), device="cuda", generator=g)
    y = torch.randint(0, VOCAB_SIZE, (1, ctx), device="cuda", generator=g)
    fwd_model = torch.compile(model) if impl == "compiled" else model

    rows = []

    def fwd():
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            return fwd_model(x)

    def fwd_bwd():
        opt.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = fwd_model(x)
            loss = torch.nn.functional.cross_entropy(
                logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
            )
        loss.backward()
        return loss

    def train_step():
        loss = fwd_bwd()
        opt.step()
        return loss

    cold = {}
    try:
        for name, fn in [("forward", fwd), ("fwd_bwd", fwd_bwd), ("train_step", train_step)]:
            if impl == "compiled":
                torch._dynamo.reset()
                fwd_model = torch.compile(model)
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                fn()
                torch.cuda.synchronize()
                cold[name] = round(time.perf_counter() - t0, 3)
            for _ in range(3):
                fn()
            q20, q50, q80 = triton.testing.do_bench(
                fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
            )
            torch.cuda.reset_peak_memory_stats()
            fn()
            torch.cuda.synchronize()
            rows.append(
                {
                    "experiment": "small_model",
                    "seq_len": ctx,
                    "head_dim": "",
                    "implementation": impl,
                    "phase": name,
                    "cold_start_s": cold.get(name),
                    "p20_ms": q20,
                    "p50_ms": q50,
                    "p80_ms": q80,
                    "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                    "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
                    "status": "ok",
                }
            )
    except torch.cuda.OutOfMemoryError:
        rows.append(
            {
                "experiment": "small_model",
                "seq_len": ctx,
                "head_dim": "",
                "implementation": impl,
                "phase": "unknown",
                "cold_start_s": None,
                "p20_ms": None,
                "p50_ms": None,
                "p80_ms": None,
                "peak_allocated_mib": "",
                "peak_reserved_mib": "",
                "status": "oom",
            }
        )
    del model, opt
    torch.cuda.empty_cache()
    torch._dynamo.reset()
    return rows


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for seq, d in ATTN_CONFIGS:
        for impl in ["eager", "compiled"]:
            r = bench_attention(seq, d, impl)
            print("attn", seq, d, impl, r[0]["status"], r[0]["p50_ms"], "cold", r[0]["cold_start_s"])
            rows.extend(r)
    for impl in ["eager", "compiled"]:
        r = bench_model(impl)
        for row in r:
            print("model", impl, row["phase"], row["status"], row["p50_ms"], "cold", row["cold_start_s"])
        rows.extend(r)

    csv_path = os.path.join(OUT_DIR, "compile_comparison.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_json(
        collect_metadata(
            {
                "script": "student_scripts/a2k/run_compile_comparison.py",
                "command": "PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 .venv/bin/python student_scripts/a2k/run_compile_comparison.py",
                "commit": git_commit(),
                "seed": SEED,
                "do_bench": {"warmup_ms": 100, "rep_ms": 300, "quantiles": [0.2, 0.5, 0.8]},
                "torch_compile_mode": "default",
            }
        ),
        os.path.join(OUT_DIR, "run_metadata_compile.json"),
    )
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
