import os
import sys
import pandas as pd
import argparse
import json
import statistics
import time
from typing import Dict, Tuple

import torch
import torch.nn as nn

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from env_utils import public_env

# pdf中的数据
MODEL_CONFIGS = {
    "small":  {"d_model": 768,  "d_ff": 3072, "num_layers": 12, "num_heads": 12},
    "medium": {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16},
    "large":  {"d_model": 1280, "d_ff": 5120, "num_layers": 36, "num_heads": 20},
    "xl":     {"d_model": 2560, "d_ff": 10240,"num_layers": 32, "num_heads": 32},
    "10B":    {"d_model": 4608, "d_ff": 12288,"num_layers": 50, "num_heads": 36},
}

VOCAB_SIZE = 10000

def get_model(model_size: str, context_length: int, dtype: torch.dtype) -> nn.Module:
    config = MODEL_CONFIGS[model_size]
    model = BasicsTransformerLM(
        vocab_size=VOCAB_SIZE,
        context_length=context_length,
        d_model=config["d_model"],
        num_layers=config["num_layers"],
        num_heads=config["num_heads"],
        d_ff=config["d_ff"],
        rope_theta=10000.0,   # RoPE 的 theta 参数，使用默认值
    )
    # 转换模型精度并移动到 GPU
    model = model.to(dtype=dtype)
    model = model.cuda()
    return model

def get_optimizer(model: nn.Module) -> torch.optim.Optimizer:
    return AdamW(model.parameters(), lr=1e-3)

def generate_batch(batch_size: int, context_length: int, device: torch.device):
    """
    生成随机的输入 token IDs 和标签（均为整数张量）。

    参数:
        batch_size: 批量大小。
        context_length: 序列长度。
        device: 目标设备（如 'cuda'）。

    返回:
        (input_ids, labels) 元组，形状均为 (batch_size, context_length)。
    """
    input_ids = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    labels = torch.randint(0, VOCAB_SIZE, (batch_size, context_length), device=device)
    return input_ids, labels

def run_step(
    mode: str,
    model: nn.Module,
    batch: Tuple[torch.Tensor, torch.Tensor],
    optimizer: torch.optim.Optimizer | None = None,
) -> None:
    """
    根据指定模式执行一个步骤（不返回时间，仅执行操作）。

    参数:
        mode: 执行模式，'forward'、'forward_backward' 或 'train_step'。
        model: 模型。
        batch: (input_ids, labels) 元组。
        optimizer: 优化器（仅在 train_step 模式时需要）。
    """
    input_ids, labels = batch

    if mode == "forward":
        # 仅前向传播，使用 no_grad 禁用梯度计算以节省显存和计算
        with torch.no_grad():
            model(input_ids)

    elif mode == "forward_backward":
        # 前向 + 反向，但不更新参数
        model.zero_grad(set_to_none=True)          # 清空梯度
        logits = model(input_ids)                  # 前向
        loss = cross_entropy(logits, labels)       # 计算损失
        loss.backward()                            # 反向传播

    elif mode == "train_step":
        # 完整训练步骤：zero_grad -> forward -> loss -> backward -> optimizer.step
        model.zero_grad(set_to_none=True)
        logits = model(input_ids)
        loss = cross_entropy(logits, labels)
        loss.backward()
        optimizer.step()                           # 更新参数

    else:
        raise ValueError(f"Unknown mode: {mode}")
    
def benchmark(args) -> Dict:
    """
    运行基准测试，返回包含原始计时和统计信息的字典。

    参数:
        args: 命令行参数（argparse.Namespace）。

    返回:
        包含模型配置、原始时间列表、均值、标准差和变异系数的字典。
    """
    device = torch.device("cuda")
    torch.manual_seed(args.seed)   # 固定随机种子，保证可重复性

    # 将 dtype 字符串映射为 torch 数据类型
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if args.dtype not in dtype_map:
        raise ValueError(f"Unsupported dtype: {args.dtype}")
    dtype = dtype_map[args.dtype]

    # 初始化模型、优化器（如果需要）和批量数据
    model = get_model(args.model_size, args.context_length, dtype)
    optimizer = get_optimizer(model) if args.mode == "train_step" else None
    batch = generate_batch(args.batch_size, args.context_length, device)

    # ---------- 预热阶段（不计时） ----------
    # 目的：让 CUDA/cuDNN 完成 kernel 自动调优、显存分配等初始化工作，
    # 避免首次运行的高开销影响测量结果。
    for _ in range(args.warmup):
        run_step(args.mode, model, batch, optimizer)
        torch.cuda.synchronize()   # 等待 GPU 完成，确保下一次运行独立

    # ---------- 测量阶段 ----------
    # 显式同步边界：确保 warm-up 阶段的所有 CUDA kernel 都已排空后再开始计时，
    # 避免上一个 warm-up 步骤的尾部内核进入第一个测量区间。
    torch.cuda.synchronize()

    timings = []
    for _ in range(args.steps):
        start = time.perf_counter()                # 高精度计时开始
        run_step(args.mode, model, batch, optimizer)  # 执行一步
        torch.cuda.synchronize()                   # 等待 GPU 完成所有内核
        end = time.perf_counter()                  # 计时结束
        timings.append(end - start)                # 记录本步骤耗时（秒）

    # 计算统计量
    mean = statistics.mean(timings)                # 平均耗时
    stdev = statistics.stdev(timings) if len(timings) > 1 else 0.0  # 样本标准差
    cv = (stdev / mean) * 100 if mean > 0 else 0.0  # 变异系数（%）

    return {
        "model_size": args.model_size,
        "batch_size": args.batch_size,
        "context_length": args.context_length,
        "dtype": args.dtype,
        "mode": args.mode,
        "warmup": args.warmup,
        "steps": args.steps,
        "timings_sec": timings,      
        "mean_sec": mean,
        "stddev_sec": stdev,
        "cv_percent": cv,
    }

def main():
    parser = argparse.ArgumentParser(description="Benchmark cs336 Transformer model")
    parser.add_argument("--model-size", type=str, default="small",
                        choices=MODEL_CONFIGS.keys(), help="模型规模")
    parser.add_argument("--batch-size", type=int, default=4, help="批量大小")
    parser.add_argument("--context-length", type=int, default=512, help="上下文长度")
    parser.add_argument("--dtype", type=str, default="fp32",
                        choices=["fp32", "fp16", "bf16"], help="数据类型")
    parser.add_argument("--mode", type=str, required=True,
                        choices=["forward", "forward_backward", "train_step"],
                        help="基准测试模式")
    parser.add_argument("--warmup", type=int, default=5, help="预热步骤数")
    parser.add_argument("--steps", type=int, default=10, help="测量步骤数")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--output", type=str, required=True, help="输出 CSV 文件路径")
    args = parser.parse_args()

    # 运行基准测试
    result = benchmark(args)

    records = []
    for i, t in enumerate(result["timings_sec"]):
        records.append({
            "model_size": result["model_size"],
            "batch_size": result["batch_size"],
            "context_length": result["context_length"],
            "dtype": result["dtype"],
            "mode": result["mode"],
            "warmup": result["warmup"],
            "steps": result["steps"],
            "timing_index": i,
            "time_sec": round(t, 6),
            "mean_sec": round(result["mean_sec"], 6),
            "stddev_sec": round(result["stddev_sec"], 6),
            "cv_percent": round(result["cv_percent"], 2),
        })
    df = pd.DataFrame(records)

    # 确保输出目录存在
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)

    # 追加写入 CSV（首次写入表头）
    file_exists = os.path.exists(args.output) and os.path.getsize(args.output) > 0
    df.to_csv(args.output, mode='a', header=not file_exists, index=False)

    # ---- 每次运行都写一份轻量 metadata（命令 + 配置 + 去敏环境版本）----
    # 与结果 CSV 同目录的 benchmark_metadata.jsonl，每行一条 JSON，可追溯。
    meta = {
        "command": sys.argv,
        "result_path": os.path.abspath(args.output),
        "config": {
            "model_size": result["model_size"],
            "batch_size": result["batch_size"],
            "context_length": result["context_length"],
            "dtype": result["dtype"],
            "mode": result["mode"],
            "warmup": result["warmup"],
            "steps": result["steps"],
            "seed": args.seed,
        },
        "mean_sec": result["mean_sec"],
        "stddev_sec": result["stddev_sec"],
        "cv_percent": result["cv_percent"],
        "env": public_env(),
    }
    meta_path = os.path.join(output_dir, "benchmark_metadata.jsonl")
    with open(meta_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta, default=str) + "\n")

    print(json.dumps(result, indent=2))
    
if __name__ == "__main__":
    main()