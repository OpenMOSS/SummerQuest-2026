from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="学习率扫")
    parser.add_argument("--lr-list", type=str, nargs="+", required=True,
                        help="学习率列表，如 1e-4 3e-4 5e-4 1e-3")
    parser.add_argument("--num-steps", type=int, default=1000,
                        help="每个 run 的训练步数")
    parser.add_argument("--out-dir", type=str, default="logs/lr_sweep")

    # 以下参数会透传给 train_lm.py
    parser.add_argument("--train-data", type=str, required=True)
    parser.add_argument("--val-data", type=str, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--warmup-iters", type=int, default=100)
    parser.add_argument("--device", type=str, default="cpu")

    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    results = []

    for lr in args.lr_list:
        lr_safe = lr.replace("-", "neg_").replace(".", "p")
        run_dir = out_dir / f"lr_{lr_safe}"
        run_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable, str(Path(__file__).parent / "train_lm.py"),
            "--train-data", args.train_data,
            "--val-data", args.val_data,
            "--vocab-size", str(args.vocab_size),
            "--context-length", str(args.context_length),
            "--d-model", str(args.d_model),
            "--d-ff", str(args.d_ff),
            "--num-layers", str(args.num_layers),
            "--num-heads", str(args.num_heads),
            "--rope-theta", str(args.rope_theta),
            "--batch-size", str(args.batch_size),
            "--num-steps", str(args.num_steps),
            "--lr", lr,
            "--warmup-iters", str(args.warmup_iters),
            "--cosine-cycle-iters", str(args.num_steps),
            "--out-dir", str(run_dir),
            "--log-name", "train_log.jsonl",
            "--device", args.device,
        ]

        print(f"\n{'='*60}")
        print(f"[lr_sweep] starting lr={lr}")
        print(f"[lr_sweep] output: {run_dir}")
        print(f"{'='*60}", flush=True)

        subprocess.run(cmd, check=True)

        # 读取 summary
        summary_path = run_dir / "summary.json"
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
            results.append({
                "lr": float(lr),
                "final_val_loss": summary.get("final_val_loss"),
                "total_time_sec": summary.get("total_train_time_sec"),
            })

            print(f"[lr_sweep] lr={lr} | val_loss={summary.get('final_val_loss')}", flush=True)

    # 写汇总
    sweep_summary_path = out_dir / "sweep_summary.json"
    with open(sweep_summary_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[lr_sweep] summary saved to {sweep_summary_path}", flush=True)

    # 打印表格
    print(f"\n{'='*60}")
    print(f"{'LR':>10} | {'Val Loss':>10} | {'Time (s)':>10}")
    print(f"{'-'*10}-+-{'-'*10}-+-{'-'*10}")
    for r in results:
        vl = f"{r['final_val_loss']:.4f}" if r['final_val_loss'] is not None else "N/A"
        t = f"{r['total_time_sec']:.0f}" if r['total_time_sec'] is not None else "N/A"
        print(f"{r['lr']:>10} | {vl:>10} | {t:>10}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
