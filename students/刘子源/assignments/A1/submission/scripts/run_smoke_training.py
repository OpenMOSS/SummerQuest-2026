# 十步冒烟，只查链路通不通。这个 loss 没业务含义，别拿去写报告。

from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.training import train  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_root = Path(tmp_dir)
        # 随机假数据就够了，省得跑冒烟还要先下载一坨语料。
        rand = np.random.default_rng(0)
        np.save(tmp_root / "train.npy", rand.integers(0, 64, 4096, dtype=np.uint16))
        np.save(tmp_root / "valid.npy", rand.integers(0, 64, 1024, dtype=np.uint16))
        run_output = tmp_root / "output"
        summary = train(
            {
                "run_name": "synthetic_smoke",
                "dataset": "deterministic_synthetic_tokens",
                "train_data": str(tmp_root / "train.npy"),
                "validation_data": str(tmp_root / "valid.npy"),
                "output_dir": str(run_output),
                "vocab_size": 64,
                "context_length": 16,
                "d_model": 32,
                "d_ff": 64,
                "num_layers": 2,
                "num_heads": 4,
                "rope_theta": 10000.0,
                "batch_size": 4,
                "total_steps": 10,
                "max_learning_rate": 0.001,
                "min_learning_rate": 0.0001,
                "warmup_steps": 2,
                "cosine_cycle_steps": 10,
                "weight_decay": 0.01,
                "max_grad_norm": 1.0,
                "log_interval": 1,
                "eval_interval": 2,
                "eval_batches": 2,
                "checkpoint_interval": 0,
                "seed": 0,
                "device": "cpu",
            }
        )
        shutil.copy2(run_output / "train.jsonl", args.output_dir / "smoke_train.jsonl")
        summary.pop("checkpoint", None)
        summary["log"] = "logs/smoke_train.jsonl"
        summary["scope"] = "smoke_test_only_not_a_required_TinyStories_or_OWT_result"
        (args.output_dir / "smoke_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
