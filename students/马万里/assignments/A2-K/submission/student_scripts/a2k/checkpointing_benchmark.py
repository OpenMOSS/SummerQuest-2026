"""A2-K 任务一：测量 Transformer activation checkpointing。"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import statistics
import time
from pathlib import Path

import torch

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from cs336_systems.a2k.checkpointing import forward_with_checkpoint

MODEL_CONFIG = {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16}
VOCAB_SIZE = 10_000


def set_allocator_limit() -> float:
    """在任何 CUDA 张量创建前设置 23 GiB allocator 上限。"""
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    limit_bytes = 23 * 1024**3
    fraction = min(1.0, limit_bytes / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def make_model(context_length: int) -> BasicsTransformerLM:
    """创建保持 FP32 参数的 medium 模型。"""
    return BasicsTransformerLM(vocab_size=VOCAB_SIZE, context_length=context_length, rope_theta=10_000.0, **MODEL_CONFIG).cuda()


def train_step(model, optimizer, tokens, labels, block_size, autocast_dtype):
    """完成一个完整训练 step，计时边界不包含数据和模型创建。"""
    optimizer.zero_grad(set_to_none=True)
    with torch.autocast(device_type="cuda", dtype=autocast_dtype):
        logits = forward_with_checkpoint(model, tokens, block_size)
        loss = cross_entropy(logits, labels)
    loss.backward()
    optimizer.step()


def run_one(context_length: int, block_size: int | None, warmup: int, steps: int, seed: int):
    """运行单个配置并返回可序列化的轻量结果。"""
    torch.manual_seed(seed)
    model = make_model(context_length)
    optimizer = AdamW(model.parameters(), lr=1e-3)
    tokens = torch.randint(VOCAB_SIZE, (1, context_length), device="cuda")
    labels = torch.randint(VOCAB_SIZE, (1, context_length), device="cuda")
    for _ in range(warmup):
        train_step(model, optimizer, tokens, labels, block_size, torch.bfloat16)
        torch.cuda.synchronize()

    samples = []
    status = "ok"
    for _ in range(steps):
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.synchronize()
        begin = time.perf_counter()
        try:
            train_step(model, optimizer, tokens, labels, block_size, torch.bfloat16)
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - begin) * 1000.0)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.synchronize()
            samples.append(float("nan"))
            status = "oom"
            break
    allocated = torch.cuda.max_memory_allocated() / 2**20
    reserved = torch.cuda.max_memory_reserved() / 2**20
    finite = [value for value in samples if math.isfinite(value)]
    return {
        "config_id": f"medium_k{context_length}_{'none' if block_size is None else f'block{block_size}'}",
        "model_size": "medium", "num_layers": MODEL_CONFIG["num_layers"], "context_length": context_length,
        "batch_size": 1, "dtype": "fp32_params+bf16_autocast",
        "checkpoint_block_size": "" if block_size is None else block_size, "nested": False,
        "warmup_steps": warmup, "measurement_steps": steps,
        "step_time_ms_samples": ";".join(f"{value:.4f}" for value in samples),
        "step_time_ms_p50": f"{statistics.median(finite):.4f}" if finite else "",
        "peak_allocated_mib": f"{allocated:.3f}", "peak_reserved_mib": f"{reserved:.3f}", "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="A2-K 任务一 checkpointing 矩阵")
    parser.add_argument("--context-lengths", nargs="+", type=int, default=[1024, 2048])
    parser.add_argument("--block-sizes", nargs="+", type=int, default=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    parser.add_argument("--single-context", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--single-block", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--single-output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.single_context is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("必须在 srun 申请的 GPU 节点执行")
        # 单配置子进程也必须在第一次 CUDA allocation 前设置 allocator 上限。
        set_allocator_limit()
        block_size = None if args.single_block == 0 else args.single_block
        row = run_one(args.single_context, block_size, args.warmup, args.steps, args.seed)
        args.single_output.parent.mkdir(parents=True, exist_ok=True)
        exists = args.single_output.exists() and args.single_output.stat().st_size > 0
        with args.single_output.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(row))
            if not exists:
                writer.writeheader()
            writer.writerow(row)
        return
    if not torch.cuda.is_available():
        raise RuntimeError("任务一正式测量需要 CUDA，请通过 srun 申请 RTX 4090 节点")
    fraction = set_allocator_limit()
    rows = []
    for context_length in args.context_lengths:
        for block_size in [None, *args.block_sizes]:
            rows.append(run_one(context_length, block_size, args.warmup, args.steps, args.seed))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)
    metadata = {"allocator_limit_mib": 23552, "allocator_fraction": fraction, "seed": args.seed, "command": os.sys.argv, "cuda_device": torch.cuda.get_device_name(0)}
    args.output.with_name("checkpointing_metadata.json").write_text(json.dumps(metadata, ensure_ascii=True, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
