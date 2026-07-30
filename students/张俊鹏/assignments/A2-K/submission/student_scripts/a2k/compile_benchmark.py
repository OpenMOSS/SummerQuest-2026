from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
for import_root in (REPO_ROOT, REPO_ROOT / "cs336-basics"):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW


SMALL_CONFIG = {"d_model": 768, "d_ff": 3072, "num_layers": 12, "num_heads": 12}
ALLOCATOR_LIMIT_MIB = 23 * 1024


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="A2-K torch.compile small-model benchmark")
    parser.add_argument("--implementation", choices=("eager", "compiled"), required=True)
    parser.add_argument("--phase", choices=("forward", "forward_backward", "training_step"), required=True)
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def configure_allocator_limit() -> float:
    total_bytes = torch.cuda.get_device_properties(0).total_memory
    fraction = min(1.0, ALLOCATOR_LIMIT_MIB * 1024**2 / total_bytes)
    torch.cuda.set_per_process_memory_fraction(fraction, device=0)
    return fraction


def main() -> int:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if args.warmup_steps < 0 or args.measurement_steps <= 0:
        raise ValueError("warmup steps must be non-negative and measurements positive")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    allocator_fraction = configure_allocator_limit()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch._dynamo.reset()

    row: dict[str, object] = {
        "implementation": args.implementation,
        "model_size": "small",
        "batch_size": 1,
        "context_length": 512,
        "dtype": "bf16-autocast/fp32-params",
        "phase": args.phase,
        "warmup_steps": args.warmup_steps,
        "measurement_steps": args.measurement_steps,
        "allocator_limit_mib": ALLOCATOR_LIMIT_MIB,
        "allocator_fraction": allocator_fraction,
        "compile_backend": "inductor" if args.implementation == "compiled" else "",
        "shape_specialization": "batch=1,context=512",
        "status": "ok",
        "error": "",
    }
    try:
        model = BasicsTransformerLM(
            vocab_size=10_000, context_length=512, rope_theta=10_000.0, **SMALL_CONFIG
        ).to(device="cuda", dtype=torch.float32)
        if args.implementation == "compiled":
            model = torch.compile(model, backend="inductor", fullgraph=False)
        optimizer = AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, 10_000, (1, 512), device="cuda")
        targets = torch.randint(0, 10_000, (1, 512), device="cuda")

        def run_once() -> None:
            if args.phase == "forward":
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    model(tokens)
            else:
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    logits = model(tokens)
                    loss = cross_entropy(logits, targets)
                loss.backward()
                if args.phase == "training_step":
                    optimizer.step()

        torch.cuda.synchronize()
        cold_start = time.perf_counter()
        run_once()
        torch.cuda.synchronize()
        row["cold_start_ms"] = (time.perf_counter() - cold_start) * 1_000

        for _ in range(args.warmup_steps):
            run_once()
        torch.cuda.synchronize()
        samples: list[float] = []
        allocated: list[float] = []
        reserved: list[float] = []
        for _ in range(args.measurement_steps):
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
            start = time.perf_counter()
            run_once()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1_000)
            allocated.append(torch.cuda.max_memory_allocated() / 1024**2)
            reserved.append(torch.cuda.max_memory_reserved() / 1024**2)
        graph_breaks = sum(torch._dynamo.utils.counters["graph_break"].values())
        row.update({
            "latency_ms_samples": json.dumps(samples),
            "latency_ms_p50": statistics.median(samples),
            "peak_allocated_mib": max(allocated),
            "peak_reserved_mib": max(reserved),
            "graph_break_count": graph_breaks if args.implementation == "compiled" else 0,
        })
    except Exception as exc:
        row.update({
            "status": "compile_failed" if args.implementation == "compiled" else "failed",
            "error": f"{type(exc).__name__}: {exc}",
            "cold_start_ms": "",
            "latency_ms_samples": "[]",
            "latency_ms_p50": "",
            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 1024**2,
            "peak_reserved_mib": torch.cuda.max_memory_reserved() / 1024**2,
            "graph_break_count": "",
        })
    with args.output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)
    print(json.dumps(row, ensure_ascii=False))
    return 0 if row["status"] == "ok" else 1


if __name__ == "__main__":
    raise SystemExit(main())
