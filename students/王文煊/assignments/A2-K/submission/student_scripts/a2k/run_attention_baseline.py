"""A2-K task 2 (section 6.1): explicit PyTorch attention baseline benchmark.

batch 1, BF16, causal, seq {512, 2048, 8192} x d {64, 128}, phases
forward / backward / forward+backward, measured with
triton.testing.do_bench(warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]).

Run:
    PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python student_scripts/a2k/run_attention_baseline.py
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

OUT_DIR = os.path.join("local_results", "a2k")
SEQS = [512, 2048, 8192]
DS = [64, 128]
PHASES = ["forward", "backward", "fwd_bwd"]
SEED = 0


def bench_one(seq: int, d: int, phase: str) -> dict:
    torch.manual_seed(SEED)
    q = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    k = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    v = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    do = torch.randn(1, seq, d, device="cuda", dtype=torch.bfloat16)

    def fwd():
        return explicit_attention(q, k, v, is_causal=True)

    def bwd():
        o = fwd()
        o.backward(do, retain_graph=True)
        for t in (q, k, v):
            t.grad = None

    def fwd_bwd():
        o = fwd()
        o.backward(do)
        for t in (q, k, v):
            t.grad = None

    fn = {"forward": fwd, "backward": bwd, "fwd_bwd": fwd_bwd}[phase]
    status = "ok"
    q20 = q50 = q80 = None
    peak_alloc = peak_resv = ""
    try:
        if phase == "forward":
            with torch.no_grad():
                q20, q50, q80 = triton.testing.do_bench(
                    fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
                )
                torch.cuda.reset_peak_memory_stats()
                fn()
        else:
            q20, q50, q80 = triton.testing.do_bench(
                fn, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8]
            )
            torch.cuda.reset_peak_memory_stats()
            fn()
        torch.cuda.synchronize()
        peak_alloc = mib(torch.cuda.max_memory_allocated())
        peak_resv = mib(torch.cuda.max_memory_reserved())
    except torch.cuda.OutOfMemoryError:
        status = "oom"
    row = {
        "implementation": "explicit_pytorch",
        "seq_len": seq,
        "head_dim": d,
        "batch_size": 1,
        "dtype": "bf16",
        "is_causal": True,
        "phase": phase,
        "p20_ms": q20,
        "p50_ms": q50,
        "p80_ms": q80,
        "peak_allocated_mib": peak_alloc,
        "peak_reserved_mib": peak_resv,
        "status": status,
    }
    del q, k, v, do
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for seq in SEQS:
        for d in DS:
            for phase in PHASES:
                row = bench_one(seq, d, phase)
                print(row["seq_len"], row["head_dim"], phase, row["status"], row["p50_ms"])
                rows.append(row)
    csv_path = os.path.join(OUT_DIR, "attention_baseline.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    write_json(
        collect_metadata(
            {
                "script": "student_scripts/a2k/run_attention_baseline.py",
                "command": "PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 .venv/bin/python student_scripts/a2k/run_attention_baseline.py",
                "commit": git_commit(),
                "seed": SEED,
                "do_bench": {"warmup_ms": 100, "rep_ms": 300, "quantiles": [0.2, 0.5, 0.8]},
            }
        ),
        os.path.join(OUT_DIR, "run_metadata_attention_baseline.json"),
    )
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
