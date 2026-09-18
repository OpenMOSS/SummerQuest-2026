"""Memory snapshot script for A2-P task 4.

Warm-up first, then enable torch.cuda.memory._record_memory_history(),
run the requested mode, and _dump_snapshot() a pickle for the PyTorch
memory visualizer. Also prints active/allocated/reserved/peak stats with
distinct accounting semantics. Snapshot pickles stay in local results/ dir.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import autocast_ctx, build_model, collect_metadata, make_batch, save_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="xl")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=128)
    p.add_argument("--mode", choices=["forward", "train_step"], required=True)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--steps", type=int, default=3, help="steps recorded inside history")
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"

    model = build_model(args.model_size, args.context_length).to(device)
    x, y = make_batch(args.batch_size, args.context_length, device, args.seed)

    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    optimizer = AdamW(model.parameters(), lr=args.lr) if args.mode == "train_step" else None

    def step():
        if args.mode == "forward":
            with torch.no_grad(), autocast_ctx(args.dtype):
                model(x)
        else:
            optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(args.dtype):
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            optimizer.step()

    # warm-up BEFORE enabling memory history
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    torch.cuda.memory._record_memory_history(max_entries=1_000_000)
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        step()
        torch.cuda.synchronize()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_size}_ctx{args.context_length}_bs{args.batch_size}_{args.mode}_{args.dtype}"
    snap_path = out_dir / f"snapshot_{tag}.pickle"
    torch.cuda.memory._dump_snapshot(str(snap_path))
    torch.cuda.memory._record_memory_history(enabled=None)

    stats = torch.cuda.memory_stats()
    mem = torch.cuda.memory_stats()
    summary = {
        "metadata": collect_metadata(
            command="python " + " ".join(sys.argv),
            config=vars(args),
            results_path=str(snap_path),
        ),
        # distinct accounting semantics:
        "active_bytes_now": torch.cuda.memory_allocated(),          # live tensors right now
        "peak_allocated_bytes": torch.cuda.max_memory_allocated(),  # peak active bytes
        "reserved_bytes_now": torch.cuda.memory_reserved(),         # caching-allocator pool
        "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
        "num_alloc_retries": stats.get("num_alloc_retries", 0),
    }
    save_json(summary, out_dir / f"memsummary_{tag}.json")
    print(f"snapshot -> {snap_path}")
    for k in ("active_bytes_now", "peak_allocated_bytes", "reserved_bytes_now", "peak_reserved_bytes"):
        print(f"  {k}: {summary[k]/2**30:.3f} GiB")


if __name__ == "__main__":
    main()
