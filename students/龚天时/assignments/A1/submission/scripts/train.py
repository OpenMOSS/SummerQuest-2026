"""
  uv run python scripts/train.py --train_data data/ts_train.npy --val_data data/ts_valid.npy \
      --total_steps 5000 --batch_size 32 --d_model 512 --num_layers 4
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.training import (
    cross_entropy,
    AdamW,
    get_lr_cosine_schedule,
    gradient_clipping,
    get_batch,
    save_checkpoint,
    load_checkpoint,
)


@torch.no_grad()                         
def evaluate(model, data, batch_size, context_length, vocab_size, device, n_batches):
    model.eval()                         
    losses = []
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits.view(-1, vocab_size), y.view(-1))
        losses.append(loss.item())
    model.train()                         
    return sum(losses) / len(losses)


def main():
    # ────────── 命令行参数 ──────────
    p = argparse.ArgumentParser()
    # 数据
    p.add_argument("--train_data", type=str, required=True)
    p.add_argument("--val_data", type=str, required=True)
    # 模型
    p.add_argument("--vocab_size", type=int, default=10000)
    p.add_argument("--context_length", type=int, default=256)
    p.add_argument("--d_model", type=int, default=512)
    p.add_argument("--num_layers", type=int, default=4)
    p.add_argument("--num_heads", type=int, default=16)
    p.add_argument("--d_ff", type=int, default=1344)
    p.add_argument("--rope_theta", type=float, default=10000.0)
    # 优化器 + 学习率
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--warmup_iters", type=int, default=200)
    p.add_argument("--grad_clip", type=float, default=1.0)
    # 训练
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--total_steps", type=int, default=5000)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    # 日志 / checkpoint
    p.add_argument("--out_dir", type=str, default="runs/default")
    p.add_argument("--log_interval", type=int, default=10)
    p.add_argument("--eval_interval", type=int, default=200)
    p.add_argument("--eval_batches", type=int, default=20)
    p.add_argument("--ckpt_interval", type=int, default=1000)
    p.add_argument("--resume", type=str, default=None)
    # 消融
    p.add_argument("--use_rmsnorm", type=str, default="true", choices=["true", "false"])
    p.add_argument("--use_rope", type=str, default="true", choices=["true", "false"])
    p.add_argument("--norm_position", type=str, default="pre", choices=["pre", "post"])
    p.add_argument("--ffn_type", type=str, default="swiglu", choices=["swiglu", "silu"])
    args = p.parse_args()

    use_rmsnorm = args.use_rmsnorm == "true"
    use_rope = args.use_rope == "true"

    device = torch.device(args.device)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w") as f:
        json.dump(vars(args), f, indent=2)

    train_data = np.load(args.train_data, mmap_mode="r")
    val_data = np.load(args.val_data, mmap_mode="r")


    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=use_rmsnorm,         
        norm_position=args.norm_position,  
        use_rope=use_rope,                 
        ffn_type=args.ffn_type,           
    ).to(device)

    optimizer = AdamW(
        model.parameters(),
        lr=args.max_lr,
        weight_decay=args.weight_decay,
    )

    n_params = sum(p.numel() for p in model.parameters())
    print(f"模型参数量: {n_params:,}  |  device: {device}")

    # 恢复
    start_step = 0
    if args.resume is not None:
        start_step = load_checkpoint(args.resume, model, optimizer)
        print(f"从 {args.resume} 恢复,起始步 {start_step}")

    # 日志
    log_path = out_dir / "train_log.jsonl"
    log_f = open(log_path, "a")

    def log(record):
        log_f.write(json.dumps(record) + "\n")
        log_f.flush()                     # 立刻落盘,长任务能实时看到

    # 训练循环
    model.train()
    t0 = time.time()
    final_val_loss = None

    for step in range(start_step, args.total_steps):
        # 1. cosine schedule 算这一步的 lr,塞进优化器
        lr = get_lr_cosine_schedule(
            step, args.max_lr, args.min_lr, args.warmup_iters, args.total_steps
        )
        for g in optimizer.param_groups:
            g["lr"] = lr

        # 2. 取一批训练数据
        x, y = get_batch(train_data, args.batch_size, args.context_length, device)

        # 3. 前向 + loss
        logits = model(x)                                       # (B, T, vocab)
        loss = cross_entropy(logits.view(-1, args.vocab_size), y.view(-1))

        # 4. 反向
        optimizer.zero_grad()
        loss.backward()

        # 5. 梯度裁剪
        gradient_clipping(model.parameters(), args.grad_clip)

        # 6. 更新
        optimizer.step()

        # 7. 训练日志(题面字段:step / wall_clock_sec / train_loss / lr)
        if step % args.log_interval == 0:
            log({
                "step": step,
                "wall_clock_sec": time.time() - t0,
                "train_loss": loss.item(),
                "lr": lr,
            })
            print(f"step {step:6d} | loss {loss.item():.4f} | lr {lr:.2e} | "
                  f"{time.time()-t0:.1f}s")

        # 8. 定期验证 + checkpoint
        if step > 0 and step % args.eval_interval == 0:
            val_loss = evaluate(model, val_data, args.batch_size,
                                args.context_length, args.vocab_size,
                                device, args.eval_batches)
            log({"step": step, "val_loss": val_loss,
                 "wall_clock_sec": time.time() - t0})
            print(f"step {step:6d} | val_loss {val_loss:.4f}")

        if step > 0 and step % args.ckpt_interval == 0:
            save_checkpoint(model, optimizer, step, out_dir / "ckpt_latest.pt")
            print(f"已保存 checkpoint (step {step})")

    # 最终验证 + summary + checkpoint
    final_val_loss = evaluate(model, val_data, args.batch_size,
                              args.context_length, args.vocab_size,
                              device, args.eval_batches)
    total_time = time.time() - t0
    print(f"\n训练完成。final val_loss = {final_val_loss:.4f}  |  用时 {total_time/60:.1f} min")

    save_checkpoint(model, optimizer, args.total_steps, out_dir / "ckpt_final.pt")

    # summary
    summary = {
        "final_val_loss": final_val_loss,
        "total_train_time_sec": total_time,
        "config": {
            "d_model": args.d_model,
            "num_layers": args.num_layers,
            "num_heads": args.num_heads,
            "context_length": args.context_length,
            "batch_size": args.batch_size,
            "total_steps": args.total_steps,
        },
    }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    log_f.close()


if __name__ == "__main__":
    main()