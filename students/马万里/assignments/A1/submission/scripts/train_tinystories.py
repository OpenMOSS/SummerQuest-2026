import argparse
import json
import os
import time
import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.optim import AdamW, get_lr_cosine_schedule, clip_gradients, cross_entropy
from cs336_basics.training import get_batch, save_checkpoint, load_checkpoint


def evaluate(model, val_data, args, device):
    """在验证集上评估平均损失"""
    model.eval()
    total_loss = 0.0
    num_batches = 0
    with torch.no_grad():
        for _ in range(args.val_num_batches):
            x, y = get_batch(val_data, args.batch_size, args.context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))
            total_loss += loss.item()
            num_batches += 1
    model.train()
    return total_loss / num_batches


def train(args):
    # 确保输出目录和日志目录存在
    os.makedirs(args.out_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)

    device = args.device
    print(f"Using device: {device}")

    # 加载数据（使用 mmap 模式，避免一次性读入内存）
    train_data = np.load(args.train_data_path, mmap_mode="r")
    val_data = np.load(args.val_data_path, mmap_mode="r")
    print(f"Train tokens: {len(train_data)}, Val tokens: {len(val_data)}")

    # 构建模型
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        theta=args.theta,
        block_type=args.block_type,
        use_silu_ffn=args.use_silu_ffn,
    ).to(device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=args.weight_decay,
    )

    # 学习率调度相关
    warmup_iters = args.warmup_iters
    cosine_cycle_iters = args.total_steps

    # 恢复训练
    start_iter = 0
    if args.resume_from:
        start_iter = load_checkpoint(args.resume_from, model, optimizer)
        print(f"Resumed from iteration {start_iter}")

    # 日志记录
    log_path = os.path.join(args.log_dir, args.log_name)
    log_file = open(log_path, "w")
    start_time = time.time()

    model.train()
    for it in range(start_iter, args.total_steps):
        # 获取一个 batch
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # 前向传播
        logits = model(x)
        loss = cross_entropy(logits.reshape(-1, args.vocab_size), y.reshape(-1))

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        clip_gradients(model.parameters(), args.grad_clip)

        # 更新学习率
        lr = get_lr_cosine_schedule(it, args.max_lr, args.min_lr, warmup_iters, cosine_cycle_iters)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        # 优化器更新
        optimizer.step()

        # 打印日志
        if it % args.log_interval == 0:
            elapsed = time.time() - start_time
            log_entry = {
                "step": it,
                "train_loss": loss.item(),
                "lr": lr,
                "wall_clock_sec": elapsed,
            }
            print(f"Step {it}/{args.total_steps} | loss {loss.item():.4f} | lr {lr:.6f} | time {elapsed:.1f}s")
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

        # 验证
        if it % args.val_interval == 0 or it == args.total_steps - 1:
            val_loss = evaluate(model, val_data, args, device)
            elapsed = time.time() - start_time
            val_entry = {
                "step": it,
                "val_loss": val_loss,
                "wall_clock_sec": elapsed,
            }
            print(f"Step {it} | val_loss {val_loss:.4f}")
            log_file.write(json.dumps(val_entry) + "\n")
            log_file.flush()

        # 保存 checkpoint
        if it % args.save_interval == 0 and it > 0:
            ckpt_path = os.path.join(args.out_dir, f"checkpoint_{it}.pt")
            save_checkpoint(model, optimizer, it, ckpt_path)
            print(f"Saved checkpoint to {ckpt_path}")

    # 最终保存
    final_path = os.path.join(args.out_dir, "final.pt")
    save_checkpoint(model, optimizer, args.total_steps, final_path)
    print(f"Training finished. Final checkpoint saved to {final_path}")

    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Transformer LM on TinyStories")
    parser.add_argument("--train_data_path", type=str, required=True, help="Path to train tokens .npy")
    parser.add_argument("--val_data_path", type=str, required=True, help="Path to val tokens .npy")
    parser.add_argument("--out_dir", type=str, required=True, help="Directory to save checkpoints")
    parser.add_argument("--log_dir", type=str, default="log", help="Directory to save logs")

    # 模型参数
    parser.add_argument("--vocab_size", type=int, default=10000)
    parser.add_argument("--context_length", type=int, default=256)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--num_layers", type=int, default=4)
    parser.add_argument("--num_heads", type=int, default=16)
    parser.add_argument("--d_ff", type=int, default=1344)
    parser.add_argument("--theta", type=float, default=10000.0)

    # 训练参数
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--total_steps", type=int, default=1000)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--max_lr", type=float, default=1e-3)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--warmup_iters", type=int, default=100)
    parser.add_argument("--grad_clip", type=float, default=1.0)

    # 日志与保存
    parser.add_argument("--log_interval", type=int, default=10)
    parser.add_argument("--val_interval", type=int, default=100)
    parser.add_argument("--save_interval", type=int, default=500)
    parser.add_argument("--val_num_batches", type=int, default=20, help="Number of validation batches per eval")

    # 其他
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--resume_from", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--block_type", type=str, default="pre_norm",
                    choices=["pre_norm", "post_norm", "no_rmsnorm", "nope"])
    parser.add_argument("--use_silu_ffn", action="store_true")
    parser.add_argument("--log_name", type=str, default="train_log.jsonl",
                    help="Name of the log file (relative to log_dir)")

    args = parser.parse_args()
    train(args)