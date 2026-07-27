"""torch.profiler trace of one stable train_step with stage annotations (A2-P task 2).

Uses torch.profiler with CPU+CUDA activities and record_function ranges:
  profile/warmup, profile/measure, forward, backward, optimizer,
  attention/scores, attention/softmax, attention/value.

Attention sub-stages are annotated by monkeypatching
cs336_basics.model.scaled_dot_product_attention with an annotated wrapper.
A short schedule captures exactly one warmed-up measurement step and exports
a Chrome trace (kept in the local results/ work dir, never submitted).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import torch
import torch.profiler as profiler
from torch.profiler import record_function

sys.path.insert(0, str(Path(__file__).parent))
from common import autocast_ctx, build_model, collect_metadata, make_batch, save_json  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", default="small")
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--context-length", type=int, default=512)
    p.add_argument("--warmup", type=int, default=5)
    p.add_argument("--dtype", choices=["fp32", "bf16"], default="fp32")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--output-dir", required=True)
    return p.parse_args()


def install_annotated_attention():
    import cs336_basics.model as m
    from cs336_basics.nn_utils import softmax
    from einops import einsum

    def annotated_scaled_dot_product_attention(Q, K, V, mask=None):
        d_k = K.shape[-1]
        with record_function("attention/scores"):
            attention_scores = einsum(
                Q, K, "... query d_k, ... key d_k -> ... query key"
            ) / math.sqrt(d_k)
            if mask is not None:
                attention_scores = torch.where(mask, attention_scores, float("-inf"))
        with record_function("attention/softmax"):
            attention_weights = softmax(attention_scores, dim=-1)
        with record_function("attention/value"):
            out = einsum(
                attention_weights, V, "... query key, ... key d_v -> ... query d_v"
            )
        return out

    m.scaled_dot_product_attention = annotated_scaled_dot_product_attention


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = "cuda"
    install_annotated_attention()

    model = build_model(args.model_size, args.context_length).to(device)
    x, y = make_batch(args.batch_size, args.context_length, device, args.seed)

    from cs336_basics.nn_utils import cross_entropy
    from cs336_basics.optimizer import AdamW

    optimizer = AdamW(model.parameters(), lr=args.lr)

    def train_step():
        optimizer.zero_grad(set_to_none=True)
        with record_function("forward"), autocast_ctx(args.dtype):
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
        with record_function("backward"):
            loss.backward()
        with record_function("optimizer"):
            optimizer.step()
        return float(loss.detach().float())

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{args.model_size}_ctx{args.context_length}_bs{args.batch_size}_{args.dtype}"
    trace_path = out_dir / f"trace_{tag}.json"

    def trace_handler(prof):
        prof.export_chrome_trace(str(trace_path))

    with profiler.profile(
        activities=[profiler.ProfilerActivity.CPU, profiler.ProfilerActivity.CUDA],
        schedule=profiler.schedule(wait=0, warmup=args.warmup, active=1, repeat=1),
        on_trace_ready=trace_handler,
    ) as prof:
        for _ in range(args.warmup):
            with record_function("profile/warmup"):
                train_step()
            torch.cuda.synchronize()
            prof.step()
        with record_function("profile/measure"):
            loss = train_step()
        torch.cuda.synchronize()
        prof.step()

    # ---- lightweight summary from the recorded events ----
    key_avgs = prof.key_averages()
    top_ops = []
    for ev in sorted(key_avgs, key=lambda e: -e.device_time_total)[:40]:
        top_ops.append(
            {
                "key": ev.key,
                "calls": ev.count,
                "cpu_time_total_us": round(ev.cpu_time_total, 1),
                "cuda_time_total_us": round(ev.device_time_total, 1),
                "cpu_time_us": round(ev.cpu_time, 1),
                "cuda_time_us": round(ev.device_time, 1),
            }
        )

    def range_total_us(name):
        return sum(ev.device_time_total for ev in key_avgs if ev.key == name)

    stage_summary = {}
    for stage in [
        "forward", "backward", "optimizer",
        "attention/scores", "attention/softmax", "attention/value",
        "profile/measure",
    ]:
        matches = [ev for ev in key_avgs if ev.key == stage]
        if matches:
            stage_summary[stage] = {
                "calls": sum(ev.count for ev in matches),
                "cuda_time_total_us": round(sum(ev.device_time_total for ev in matches), 1),
                "cpu_time_total_us": round(sum(ev.cpu_time_total for ev in matches), 1),
            }

    summary = {
        "metadata": collect_metadata(
            command="python " + " ".join(sys.argv),
            config=vars(args),
            results_path=str(trace_path),
        ),
        "tool": "torch.profiler (CPU+CUDA activities, record_function ranges)",
        "loss_of_measured_step": loss,
        "stage_summary": stage_summary,
        "top_ops": top_ops,
    }
    save_json(summary, out_dir / f"summary_{tag}.json")
    print(f"trace -> {trace_path}")
    for k, v in stage_summary.items():
        print(f"  {k}: cuda_total={v['cuda_time_total_us']/1e3:.2f}ms calls={v['calls']}")


if __name__ == "__main__":
    main()
