"""Unified benchmark entry for A2-P task 1 / 3(c).

Modes: forward | forward_backward | train_step
Each measured CUDA step is followed by torch.cuda.synchronize().
Data generation and initialization are not timed.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import timeit
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).parent))
from common import (  # noqa: E402
    autocast_ctx,
    build_model,
    collect_metadata,
    make_batch,
    save_json,
)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="small")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--mode", choices=["forward", "forward_backward", "train_step"], required=True)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output", required=True)
    p.add_argument("--record-loss", action="store_true", help="record per-step loss values")
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
                logits = model(x)
        else:
            if args.mode == "forward_backward":
                model.zero_grad(set_to_none=True)
            else:
                optimizer.zero_grad(set_to_none=True)
            with autocast_ctx(args.dtype):
                logits = model(x)
                loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            loss.backward()
            if args.mode == "train_step":
                optimizer.step()
        if args.record_loss and args.mode != "forward":
            return float(loss.detach().float())

    # warmup (not timed)
    for _ in range(args.warmup):
        step()
    torch.cuda.synchronize()

    timings = []
    losses = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(args.steps):
        t0 = timeit.default_timer()
        l = step()
        torch.cuda.synchronize()
        timings.append(timeit.default_timer() - t0)
        if l is not None:
            losses.append(l)

    mean = statistics.mean(timings)
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0
    result = {
        "metadata": collect_metadata(
            command="python " + " ".join(sys.argv),
            config=vars(args),
            results_path=args.output,
        ),
        "timings_s": timings,
        "mean_s": mean,
        "stdev_s": stdev,
        "cv": stdev / mean if mean > 0 else 0.0,
        "peak_memory_bytes": torch.cuda.max_memory_allocated(),
        "losses": losses if losses else None,
    }
    save_json(result, args.output)
    print(f"mode={args.mode} mean={mean*1e3:.2f}ms stdev={stdev*1e3:.2f}ms "
          f"cv={result['cv']:.4f} peak_mem={result['peak_memory_bytes']/2**30:.2f}GiB")


if __name__ == "__main__":
    main()
