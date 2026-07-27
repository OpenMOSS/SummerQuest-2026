"""A2-K task 1 (section 5.2): activation checkpointing memory/latency matrix.

Stanford medium config, 24 layers, batch size 1, BF16 autocast, FP32 params,
AdamW, full training step. Non-nested checkpoint block sizes {none,1,2,4,8}
at context 1024, then {none, best} at context 2048.

Run:
    PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 \
        .venv/bin/python student_scripts/a2k/run_checkpointing.py
"""

from __future__ import annotations

import csv
import os
import sys

import torch

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__))))
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

OUT_DIR = os.path.join("local_results", "a2k")
WARMUP = 3
MEASURE = 5
SEED = 0


def make_model(ctx_len: int) -> BasicsTransformerLM:
    d_model, d_ff, num_layers, num_heads = MODEL_SIZES["medium"]
    torch.manual_seed(SEED)
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=ctx_len,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff,
    )
    return model.cuda().float()


def train_step(model, opt, x, y, block_size):
    """One full training step. block_size=0 means no checkpointing."""
    from torch.utils.checkpoint import checkpoint

    opt.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        h = model.token_embeddings(x)
        if block_size > 0:
            for start in range(0, len(model.layers), block_size):
                layers = model.layers[start : start + block_size]

                def run_group(hidden, layers=tuple(layers)):
                    for layer in layers:
                        hidden = layer(hidden)
                    return hidden

                h = checkpoint(run_group, h, use_reentrant=False)
        else:
            for layer in model.layers:
                h = layer(h)
        logits = model.lm_head(model.ln_final(h))
        loss = torch.nn.functional.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), y.reshape(-1)
        )
    loss.backward()
    opt.step()
    return loss


def bench_config(ctx_len: int, block_size: int) -> dict:
    model = make_model(ctx_len)
    opt = AdamW(model.parameters(), lr=1e-4)
    g = torch.Generator(device="cuda").manual_seed(SEED)
    x = torch.randint(0, VOCAB_SIZE, (1, ctx_len), device="cuda", generator=g)
    y = torch.randint(0, VOCAB_SIZE, (1, ctx_len), device="cuda", generator=g)

    status = "ok"
    samples: list[float] = []
    peak_alloc = peak_resv = 0.0
    try:
        for _ in range(WARMUP):
            train_step(model, opt, x, y, block_size)
        torch.cuda.synchronize()
        for _ in range(MEASURE):
            torch.cuda.reset_peak_memory_stats()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            train_step(model, opt, x, y, block_size)
            end.record()
            torch.cuda.synchronize()
            samples.append(round(start.elapsed_time(end), 3))
            peak_alloc = max(peak_alloc, mib(torch.cuda.max_memory_allocated()))
            peak_resv = max(peak_resv, mib(torch.cuda.max_memory_reserved()))
    except torch.cuda.OutOfMemoryError:
        status = "oom"
    samples_sorted = sorted(samples)
    p50 = samples_sorted[len(samples_sorted) // 2] if samples else None
    row = {
        "config_id": f"ctx{ctx_len}_bs{block_size if block_size > 0 else 'none'}",
        "model_size": "medium",
        "num_layers": 24,
        "context_length": ctx_len,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": block_size if block_size > 0 else "none",
        "nested": False,
        "warmup_steps": WARMUP,
        "measurement_steps": MEASURE,
        "step_time_ms_samples": samples,
        "step_time_ms_p50": p50,
        "peak_allocated_mib": peak_alloc if status == "ok" else "",
        "peak_reserved_mib": peak_resv if status == "ok" else "",
        "status": status,
    }
    del model, opt
    torch.cuda.empty_cache()
    return row


def main():
    os.makedirs(OUT_DIR, exist_ok=True)
    rows = []
    for bs in [0, 1, 2, 4, 8]:
        row = bench_config(1024, bs)
        print(row["config_id"], row["status"], row["step_time_ms_p50"], row["peak_allocated_mib"])
        rows.append(row)
    ok_rows = [r for r in rows if r["status"] == "ok" and r["context_length"] == 1024]
    best = min(ok_rows, key=lambda r: r["peak_allocated_mib"])
    best_bs = best["checkpoint_block_size"]
    best_bs = 0 if best_bs == "none" else int(best_bs)
    for bs in dict.fromkeys([0, best_bs]):
        row = bench_config(2048, bs)
        print(row["config_id"], row["status"], row["step_time_ms_p50"], row["peak_allocated_mib"])
        rows.append(row)

    fieldnames = list(rows[0].keys())
    csv_path = os.path.join(OUT_DIR, "checkpointing.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            r2 = dict(r)
            r2["step_time_ms_samples"] = ";".join(str(s) for s in r["step_time_ms_samples"])
            w.writerow(r2)
    write_json(
        collect_metadata(
            {
                "script": "student_scripts/a2k/run_checkpointing.py",
                "command": "PYTHONPATH=cs336-basics CUDA_VISIBLE_DEVICES=0 .venv/bin/python student_scripts/a2k/run_checkpointing.py",
                "commit": git_commit(),
                "seed": SEED,
                "warmup_steps": WARMUP,
                "measurement_steps": MEASURE,
            }
        ),
        os.path.join(OUT_DIR, "run_metadata_checkpointing.json"),
    )
    print("wrote", csv_path)


if __name__ == "__main__":
    main()
