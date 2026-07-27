"""A2-K task 5 (section 9.2): fixed performance matrix.

eager explicit attention vs torch.compile'd attention vs student Triton
FlashAttention-2. batch 1, BF16, causal. Core matrix: seq {512, 2048, 8192}
x d {64, 128} x {forward, backward, fwd_bwd}. Long-context edge: seq 16384,
d {64, 128}, eager vs Triton. do_bench(warmup=100, rep=300,
quantiles=[0.2, 0.5, 0.8]). Speedups are computed only vs the eager row with
identical shape/dtype/phase where both rows succeeded.

Run:
    PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python student_scripts/a2k/run_flash_benchmark.py
"""

from __future__ import annotations

import csv
import os
import sys

import torch
import triton

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import collect_metadata, git_commit, mib, set_allocator_limit, write_json  # noqa: E402

ALLOCATOR_FRACTION = set_allocator_limit()  # before any CUDA allocation

from cs336_systems.a2k.attention import explicit_attention  # noqa: E402
from cs336_systems.a2k.flash_triton import FlashAttentionTriton  # noqa: E402

OUT_DIR = os.path.join("local_results", "a2k")
SEED = 0
TRITON_TILES = {"BLOCK_M": 64, "BLOCK_N": 64, "num_warps": 4, "num_stages": 2}
PHASES = ["forward", "backward", "fwd_bwd"]

compiled_attention = torch.compile(explicit_attention)


def make_inputs(seq, d):
    torch.manual_seed(SEED)
    q = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16)
    return q, k, v, do


def bench(impl, seq, d, phase):
    q, k, v, do = make_inputs(seq, d)
    if impl == "eager":
        fn = explicit_attention
    elif impl == "compiled":
        fn = compiled_attention
    else:
        fn = FlashAttentionTriton.apply

    def fwd():
        return fn(q, k, v, True)

    def bwd():
        o = fwd()  # fresh graph each call, so no retain_graph needed
        o.backward(do)
        for t in (q, k, v):
            t.grad = None

    def fwd_bwd():
        o = fwd()
        o.backward(do)
        for t in (q, k, v):
            t.grad = None

    run = {"forward": fwd, "backward": bwd, "fwd_bwd": fwd_bwd}[phase]
    row = {
        "implementation": impl,
        "seq_len": seq,
        "head_dim": d,
        "batch_size": 1,
        "dtype": "bf16",
        "is_causal": True,
        "phase": phase,
        "p20_ms": None,
        "p50_ms": None,
        "p80_ms": None,
        "peak_allocated_mib": "",
        "peak_reserved_mib": "",
        "speedup_vs_eager": "",
        "status": "ok",
    }
    if impl == "triton":
        row.update({f"triton_{k2}": v2 for k2, v2 in TRITON_TILES.items()})
    try:
        if phase == "forward":
            with torch.no_grad():
                q20, q50, q80 = triton.testing.do_bench(
                    run, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
                )
                torch.cuda.reset_peak_memory_stats()
                run()
        else:
            if impl == "compiled":
                run()  # trigger compilation outside timing
            q20, q50, q80 = triton.testing.do_bench(
                run, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
            )
            torch.cuda.reset_peak_memory_stats()
            run()
        torch.cuda.synchronize()
        row.update(
            {
                "p20_ms": q20,
                "p50_ms": q50,
                "p80_ms": q80,
                "peak_allocated_mib": mib(torch.cuda.max_memory_allocated()),
                "peak_reserved_mib": mib(torch.cuda.max_memory_reserved()),
            }
        )
    except torch.cuda.OutOfMemoryError:
        row["status"] = "oom"
    except Exception as exc:  # keep going; record compile/kernel failures
        row["status"] = f"error:{type(exc).__name__}"
    del q, k, v, do
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    # Core matrix: all three implementations
    for seq in [512, 2048, 8192]:
        for d in [64, 128]:
            for impl in ["eager", "compiled", "triton"]:
                for phase in PHASES:
                    row = bench(impl, seq, d, phase)
                    print(seq, d, impl, phase, row["status"], row["p50_ms"], flush=True)
                    rows.append(row)
    # Long-context edge: eager vs triton (compiled optional -> skip to save time)
    for seq in [16384]:
        for d in [64, 128]:
            for impl in ["eager", "triton"]:
                for phase in PHASES:
                    row = bench(impl, seq, d, phase)
                    print(seq, d, impl, phase, row["status"], row["p50_ms"], flush=True)
                    rows.append(row)

    # speedup only when same shape/phase eager and impl both ok
    lut = {}
    for r in rows:
        lut[(r["implementation"], r["seq_len"], r["head_dim"], r["phase"])] = r
    for r in rows:
        if r["implementation"] == "eager":
            continue
        e = lut.get(("eager", r["seq_len"], r["head_dim"], r["phase"]))
        if (
            e
            and e["status"] == "ok"
            and r["status"] == "ok"
            and e["p50_ms"]
            and r["p50_ms"]
        ):
            r["speedup_vs_eager"] = round(e["p50_ms"] / r["p50_ms"], 4)

    fieldnames = [
        "implementation", "seq_len", "head_dim", "batch_size", "dtype", "is_causal",
        "phase", "p20_ms", "p50_ms", "p80_ms", "peak_allocated_mib",
        "peak_reserved_mib", "speedup_vs_eager",
        "triton_BLOCK_M", "triton_BLOCK_N", "triton_num_warps", "triton_num_stages",
        "status",
    ]
    csv_path = os.path.join(OUT_DIR, "flash_benchmark.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    write_json(
        collect_metadata(
            {
                "script": "student_scripts/a2k/run_flash_benchmark.py",
                "command": "PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 .venv/bin/python student_scripts/a2k/run_flash_benchmark.py",
                "commit": git_commit(),
                "seed": SEED,
                "do_bench": {"warmup_ms": 100, "rep_ms": 300, "quantiles": [0.2, 0.5, 0.8]},
                "triton_config": TRITON_TILES,
            }
        ),
        os.path.join(OUT_DIR, "run_metadata_flash_benchmark.json"),
    )
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
