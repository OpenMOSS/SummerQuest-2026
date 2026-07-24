from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

# 确保能 import cs336_basics
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cs336_basics.transformer import TransformerLM
from cs336_basics.optimizer import AdamW, get_lr_cosine_schedule
from cs336_basics.data import get_batch
from cs336_basics.nn_utils import gradient_clipping, cross_entropy
from cs336_basics.checkpoint import save_checkpoint, load_checkpoint


# ============================================================
# 消融变体模型
# ============================================================

class NoPETransformerBlock(nn.Module):
    """NoPE：移除 RoPE，仅靠 causal mask 提供位置信息。"""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta,
                 device=None, dtype=None):
        super().__init__()
        from cs336_basics.nn_utils import RMSNorm, SwiGLU
        from cs336_basics.transformer import MultiHeadSelfAttention
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta=None,
            device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x))
        x = x + self.ffn(self.ln2(x))
        return x


class PostNormTransformerBlock(nn.Module):
    """Post-Norm：残差后再做 RMSNorm（与 Pre-Norm 相反）。"""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta,
                 device=None, dtype=None):
        super().__init__()
        from cs336_basics.nn_utils import RMSNorm, SwiGLU
        from cs336_basics.transformer import MultiHeadSelfAttention
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta,
            device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = self.ln1(x + self.attn(x, token_positions))
        x = self.ln2(x + self.ffn(x))
        return x


class NoRMSNormBlock(nn.Module):
    """删除 RMSNorm：直接残差连接，不做归一化。"""

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta,
                 device=None, dtype=None):
        super().__init__()
        from cs336_basics.nn_utils import SwiGLU
        from cs336_basics.transformer import MultiHeadSelfAttention
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta,
            device=device, dtype=dtype,
        )
        self.ffn = SwiGLU(d_model, d_ff, device=device, dtype=dtype)

    def forward(self, x, token_positions=None):
        x = x + self.attn(x, token_positions)
        x = x + self.ffn(x)
        return x


class SiLUFFNBlock(nn.Module):
    """SiLU FFN：用 SiLU(x) 替代 SwiGLU，参数量近似匹配。

    SwiGLU 有 3 个矩阵 (w1, w2, w3)，参数量 = 3 * d_model * d_ff。
    SiLU FFN 只有 2 个矩阵 (fc1, fc2)，参数量 = 2 * d_model * d_ff_silu。
    令 d_ff_silu = 3 * d_ff / 2 可使参数量近似相等。
    """

    def __init__(self, d_model, num_heads, d_ff, max_seq_len, rope_theta,
                 device=None, dtype=None):
        super().__init__()
        from cs336_basics.nn_utils import RMSNorm, Linear, silu
        from cs336_basics.transformer import MultiHeadSelfAttention
        d_ff_silu = d_ff * 3 // 2  # 参数量近似匹配
        self.ln1 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.attn = MultiHeadSelfAttention(
            d_model, num_heads, max_seq_len, rope_theta,
            device=device, dtype=dtype,
        )
        self.ln2 = RMSNorm(d_model, eps=1e-5, device=device, dtype=dtype)
        self.fc1 = Linear(d_model, d_ff_silu, device=device, dtype=dtype)
        self.fc2 = Linear(d_ff_silu, d_model, device=device, dtype=dtype)
        self._silu = silu

    def forward(self, x, token_positions=None):
        x = x + self.attn(self.ln1(x), token_positions)
        x = x + self.fc2(self._silu(self.fc1(self.ln2(x))))
        return x


def build_model(args, device, dtype):
    """根据 ablation 参数构建模型。"""
    if args.ablation == "none":
        return TransformerLM(
            args.vocab_size, args.context_length,
            args.d_model, args.num_layers, args.num_heads, args.d_ff,
            args.rope_theta, device=device, dtype=dtype,
        )

    # 消融实验需要自定义 block
    block_cls = {
        "nope": NoPETransformerBlock,
        "post_norm": PostNormTransformerBlock,
        "no_rmsnorm": NoRMSNormBlock,
        "silu_ffn": SiLUFFNBlock,
    }[args.ablation]

    class AblationLM(nn.Module):
        def __init__(self):
            super().__init__()
            from cs336_basics.nn_utils import Embedding, RMSNorm, Linear
            self.vocab_size = args.vocab_size
            self.context_length = args.context_length
            self.token_embeddings = Embedding(
                args.vocab_size, args.d_model, device=device, dtype=dtype
            )
            self.layers = nn.ModuleList([
                block_cls(
                    args.d_model, args.num_heads, args.d_ff,
                    args.context_length, args.rope_theta,
                    device=device, dtype=dtype,
                )
                for _ in range(args.num_layers)
            ])
            # no_rmsnorm 消融不加 final norm
            if args.ablation != "no_rmsnorm":
                self.ln_final = RMSNorm(
                    args.d_model, eps=1e-5, device=device, dtype=dtype
                )
            else:
                self.ln_final = nn.Identity()
            self.lm_head = Linear(
                args.d_model, args.vocab_size, device=device, dtype=dtype
            )

        def forward(self, in_indices):
            x = self.token_embeddings(in_indices)
            T = x.size(-2)
            pos = torch.arange(
                T, device=x.device, dtype=torch.long
            ).view(1, T)
            for layer in self.layers:
                x = layer(x, pos)
            x = self.ln_final(x)
            return self.lm_head(x)

    return AblationLM()


# ============================================================
# 训练循环
# ============================================================

def evaluate(model, val_data, batch_size, context_length, device, n_eval_batches=5):
    """在验证集上计算平均 loss。"""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(n_eval_batches):
            x, y = get_batch(val_data, batch_size, context_length, device)
            logits = model(x)
            # reshape: (B*T, vocab) vs (B*T,)
            loss = cross_entropy(
                logits.view(-1, logits.size(-1)),
                y.view(-1),
            )
            losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


def train(args):
    # 确定设备
    device = torch.device(args.device)
    dtype = torch.float32

    # 加载数据（mmap 避免全量加载）
    print(f"[train] loading train data from {args.train_data}", flush=True)
    train_data = np.load(args.train_data, mmap_mode="r")
    print(f"[train] train data: {len(train_data)} tokens", flush=True)

    val_data = None
    if args.val_data:
        print(f"[train] loading val data from {args.val_data}", flush=True)
        val_data = np.load(args.val_data, mmap_mode="r")
        print(f"[train] val data: {len(val_data)} tokens", flush=True)

    # 构建模型
    print(f"[train] building model (ablation={args.ablation})", flush=True)
    model = build_model(args, device, dtype)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[train] model params: {n_params:,} ({n_params/1e6:.2f}M)", flush=True)
    model.to(device)

    # 优化器
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        betas=(args.beta1, args.beta2),
        eps=args.eps,
        weight_decay=args.weight_decay,
    )

    # 恢复 checkpoint
    start_step = 0
    if args.resume:
        ckpt_path = Path(args.out_dir) / "checkpoint.pt"
        if ckpt_path.exists():
            start_step = load_checkpoint(ckpt_path, model, optimizer)
            print(f"[train] resumed from step {start_step}", flush=True)
        else:
            print(f"[train] checkpoint not found at {ckpt_path}, starting fresh", flush=True)

    # 准备日志
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / args.log_name
    summary_path = out_dir / "summary.json"

    # 配置信息
    config = {
        "d_model": args.d_model,
        "num_layers": args.num_layers,
        "num_heads": args.num_heads,
        "d_ff": args.d_ff,
        "context_length": args.context_length,
        "batch_size": args.batch_size,
        "num_steps": args.num_steps,
        "vocab_size": args.vocab_size,
        "rope_theta": args.rope_theta,
        "lr": args.lr,
        "warmup_iters": args.warmup_iters,
        "cosine_cycle_iters": args.cosine_cycle_iters,
        "weight_decay": args.weight_decay,
        "ablation": args.ablation,
        "n_params": n_params,
    }

    # 训练循环
    model.train()
    train_start = time.time()

    # 是否追加写日志（resume 时）
    log_mode = "a" if args.resume and start_step > 0 else "w"
    log_f = open(log_path, log_mode)

    print(f"[train] starting training from step {start_step} to {args.num_steps}", flush=True)
    print(f"[train] logging to {log_path}", flush=True)

    for step in range(start_step, args.num_steps):
        step_start = time.time()

        # 学习率调度
        lr = get_lr_cosine_schedule(
            step, args.lr, args.min_lr,
            args.warmup_iters, args.cosine_cycle_iters,
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 采样 batch
        x, y = get_batch(
            train_data, args.batch_size, args.context_length, str(device),
        )

        # 前向 + 反向
        optimizer.zero_grad()
        logits = model(x)
        loss = cross_entropy(
            logits.view(-1, logits.size(-1)),
            y.view(-1),
        )
        loss.backward()

        # 梯度裁剪
        gradient_clipping(model.parameters(), args.max_grad_norm)

        # 优化器步
        optimizer.step()

        step_time = time.time() - step_start

        # 日志记录
        log_entry = {
            "step": step,
            "wall_clock_sec": time.time() - train_start,
            "train_loss": loss.item(),
            "lr": lr,
        }

        # 定期验证
        if val_data is not None and (
            step % args.val_interval == 0 or step == args.num_steps - 1
        ):
            val_loss = evaluate(
                model, val_data, args.batch_size,
                args.context_length, str(device), args.val_batches,
            )
            log_entry["val_loss"] = val_loss
            print(
                f"[train] step {step}/{args.num_steps} | "
                f"train_loss={loss.item():.4f} | val_loss={val_loss:.4f} | "
                f"lr={lr:.2e} | {step_time:.2f}s/step",
                flush=True,
            )
        else:
            print(
                f"[train] step {step}/{args.num_steps} | "
                f"train_loss={loss.item():.4f} | lr={lr:.2e} | "
                f"{step_time:.2f}s/step",
                flush=True,
            )

        log_f.write(json.dumps(log_entry) + "\n")
        log_f.flush()

        # 定期保存 checkpoint
        if (step + 1) % args.checkpoint_interval == 0 or step == args.num_steps - 1:
            ckpt_path = out_dir / "checkpoint.pt"
            save_checkpoint(model, optimizer, step + 1, ckpt_path)
            print(f"[train] checkpoint saved at step {step + 1}", flush=True)

    # 训练结束
    total_time = time.time() - train_start
    log_f.close()

    # 最终验证
    final_val_loss = None
    if val_data is not None:
        final_val_loss = evaluate(
            model, val_data, args.batch_size,
            args.context_length, str(device), args.val_batches,
        )
        print(f"[train] final val_loss={final_val_loss:.4f}", flush=True)

    # 写 summary
    summary = {
        **config,
        "final_val_loss": final_val_loss,
        "total_train_time_sec": total_time,
        "total_steps": args.num_steps,
        "total_tokens_processed": args.num_steps * args.batch_size * args.context_length,
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[train] summary saved to {summary_path}", flush=True)
    print(f"[train] total time: {total_time:.1f}s", flush=True)

    # 保存最终模型
    final_model_path = out_dir / "model.pt"
    torch.save(model.state_dict(), final_model_path)
    print(f"[train] final model saved to {final_model_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="Transformer LM 训练")

    # 数据
    parser.add_argument("--train-data", type=str, required=True,
                        help="训练数据 .npy 路径")
    parser.add_argument("--val-data", type=str, default=None,
                        help="验证数据 .npy 路径")
    parser.add_argument("--vocab-size", type=int, required=True)

    # 模型架构
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)

    # 训练
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--min-lr", type=float, default=0.0)
    parser.add_argument("--warmup-iters", type=int, default=1000)
    parser.add_argument("--cosine-cycle-iters", type=int, default=10000)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--beta1", type=float, default=0.9)
    parser.add_argument("--beta2", type=float, default=0.999)
    parser.add_argument("--eps", type=float, default=1e-8)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)

    # 验证与 checkpoint
    parser.add_argument("--val-interval", type=int, default=100)
    parser.add_argument("--val-batches", type=int, default=5)
    parser.add_argument("--checkpoint-interval", type=int, default=1000)

    # 消融
    parser.add_argument("--ablation", type=str, default="none",
                        choices=["none", "no_rmsnorm", "post_norm", "nope", "silu_ffn"],
                        help="消融实验类型")

    # 输出
    parser.add_argument("--out-dir", type=str, required=True)
    parser.add_argument("--log-name", type=str, default="train_log.jsonl")
    parser.add_argument("--device", type=str, default="cpu",
                        help="cpu / cuda / mps")
    parser.add_argument("--resume", action="store_true",
                        help="从 checkpoint 恢复训练")

    args = parser.parse_args()
    train(args)


if __name__ == "__main__":
    main()
