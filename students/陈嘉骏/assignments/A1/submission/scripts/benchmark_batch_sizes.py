from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

import torch

from cs336_basics.benchmarking import (
    BatchSizeBenchmarkResult,
    benchmark_batch_sizes,
    save_batch_size_benchmark,
)
from cs336_basics.training import load_token_dataset


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Benchmark training memory and throughput by batch size.")
    parser.add_argument("--config", default="configs/tinystories_baseline.json")
    parser.add_argument("--output-dir", default="runs/tinystories_batch_benchmark")
    parser.add_argument("--batch-sizes", type=int, nargs="+", default=[1, 2, 4, 8, 16, 32, 64, 128])
    parser.add_argument("--warmup-steps", type=int, default=5)
    parser.add_argument("--measured-steps", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--seed", type=int, default=42)
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    with open(args.config, encoding="utf-8") as f:
        configuration = json.load(f)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / "results.jsonl"
    jsonl_path.unlink(missing_ok=True)
    torch.set_float32_matmul_precision("high")

    def record_result(result: BatchSizeBenchmarkResult) -> None:
        serialized = asdict(result)
        with open(jsonl_path, "a", encoding="utf-8") as f:
            json.dump(serialized, f)
            f.write("\n")
        print(json.dumps(serialized))

    dataset = load_token_dataset(configuration["data"]["train"])
    summary = benchmark_batch_sizes(
        model_config=configuration["model"],
        dataset=dataset,
        batch_sizes=args.batch_sizes,
        device=args.device,
        warmup_steps=args.warmup_steps,
        measured_steps=args.measured_steps,
        learning_rate=args.learning_rate,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        on_result=record_result,
    )
    output_path = output_dir / "summary.json"
    save_batch_size_benchmark(summary, output_path)
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
