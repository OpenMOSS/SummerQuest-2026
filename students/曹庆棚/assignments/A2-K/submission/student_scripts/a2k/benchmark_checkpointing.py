from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint
from cs336_basics.model import BasicsTransformerLM
try:
    from cs336_basics.optimizer import AdamW as BundledAdamW
except Exception:
    BundledAdamW = torch.optim.AdamW

from .common import allocator_guard, append_csv, memory_stats, quantiles, sync


MEDIUM = dict(vocab_size=10_000, d_model=1024, num_layers=24, num_heads=16, d_ff=4096)


def main() -> None:
    p = argparse.ArgumentParser(); p.add_argument("--output", type=Path, required=True); p.add_argument("--device", default="cuda"); p.add_argument("--seqs", default="1024,2048"); p.add_argument("--block-sizes", default="0,1,2,4,8"); p.add_argument("--warmup", type=int, default=3); p.add_argument("--steps", type=int, default=5)
    args = p.parse_args(); allocator_guard(args.device); fields = ["config_id", "model_size", "num_layers", "context_length", "batch_size", "dtype", "checkpoint_block_size", "nested", "warmup_steps", "measurement_steps", "step_time_ms_samples", "step_time_ms_p50", "peak_allocated_mib", "peak_reserved_mib", "status"]
    for n in map(int, args.seqs.split(",")):
        for bsz in map(int, args.block_sizes.split(",")):
            row = {"config_id": f"medium-L24-B1-T{n}-bf16-block{bsz}", "model_size": "medium", "num_layers": 24, "context_length": n, "batch_size": 1, "dtype": "bf16", "checkpoint_block_size": bsz, "nested": False, "warmup_steps": args.warmup, "measurement_steps": args.steps, "status": "ok"}
            try:
                model = BasicsTransformerLM(context_length=n, **MEDIUM).to(args.device); opt = BundledAdamW(model.parameters(), lr=1e-4); x = torch.randint(0, MEDIUM["vocab_size"], (1, n), device=args.device)
                def step():
                    opt.zero_grad(set_to_none=True); y = model.token_embeddings(x)
                    with torch.autocast(device_type="cuda" if args.device.startswith("cuda") else "cpu", dtype=torch.bfloat16):
                        if bsz <= 0:
                            for layer in model.layers: y = layer(y)
                        else:
                            for start in range(0, len(model.layers), bsz):
                                end = min(start + bsz, len(model.layers))
                                def run_segment(inp, start=start, end=end):
                                    out = inp
                                    for layer in model.layers[start:end]: out = layer(out)
                                    return out
                                y = checkpoint(run_segment, y, use_reentrant=False)
                        y = model.ln_final(y); logits = model.lm_head(y)
                    logits.float().mean().backward(); opt.step()
                for _ in range(args.warmup): step()
                vals = []
                for _ in range(args.steps):
                    if args.device.startswith("cuda"): torch.cuda.reset_peak_memory_stats()
                    sync(args.device); import time; t0 = time.perf_counter(); step(); sync(args.device); vals.append((time.perf_counter()-t0)*1e3)
                row.update({"step_time_ms_samples": ",".join(f"{v:.3f}" for v in vals), "step_time_ms_p50": float(torch.tensor(vals).median()), **memory_stats()})
            except (RuntimeError, torch.OutOfMemoryError) as e:
                row.update({"status": "oom" if "out of memory" in str(e).lower() else "error", "error": str(e)[:160]})
            append_csv(args.output, row, fields)


if __name__ == "__main__": main()
