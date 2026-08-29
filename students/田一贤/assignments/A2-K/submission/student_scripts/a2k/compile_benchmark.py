from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM

from .common import configure_allocator_guard, measure_attention_phase, native_attention


def _attention_rows(device: torch.device, warmup: int, steps: int) -> list[dict]:
    rows = []
    for seq, dim in ((512, 64), (2048, 128), (8192, 128)):
        for implementation in ("eager", "compiled"):
            for phase in ("forward", "backward", "forward_backward"):
                base = {
                    "component": "attention",
                    "shape": f"S{seq}_D{dim}",
                    "implementation": implementation,
                    "phase": phase,
                    "compile_mode": "dynamic=False"
                    if implementation == "compiled"
                    else "eager",
                }
                if implementation == "compiled":
                    fn = torch.compile(
                        lambda q, k, v: native_attention(q, k, v, True), dynamic=False
                    )
                else:

                    def fn(q, k, v):
                        return native_attention(q, k, v, True)

                q = torch.randn(
                    1, seq, dim, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                k = torch.randn_like(q, requires_grad=True)
                v = torch.randn_like(q, requires_grad=True)
                try:
                    cold_start = None
                    if implementation == "compiled":
                        torch.cuda.synchronize(device)
                        start = time.perf_counter_ns()
                        with torch.no_grad():
                            fn(q, k, v)
                        torch.cuda.synchronize(device)
                        cold_start = (time.perf_counter_ns() - start) / 1e6
                    measured = measure_attention_phase(
                        fn, q, k, v, phase, warmup=warmup, steps=steps
                    )
                    rows.append(
                        {
                            **base,
                            "cold_start_ms": cold_start or "",
                            **measured,
                            "status": "pass",
                        }
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    rows.append(
                        {**base, "status": f"oom_or_runtime_error:{type(exc).__name__}"}
                    )
                finally:
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    return rows


def _model_rows(device: torch.device, warmup: int, steps: int) -> list[dict]:
    """Compare eager/compiled small-model entry points on the required shape."""
    rows = []
    for implementation in ("eager", "compiled"):
        torch.manual_seed(42)
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=512,
            d_model=768,
            num_layers=12,
            num_heads=12,
            d_ff=3072,
        ).to(device)
        if implementation == "compiled":
            model = torch.compile(model, dynamic=False)
        tokens = torch.randint(0, 10_000, (1, 512), device=device)
        targets = torch.randint_like(tokens, high=10_000)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        def forward():
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                model(tokens)

        def forward_backward():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                logits = model(tokens)
                loss = F.cross_entropy(logits.flatten(0, -2), targets.flatten())
            loss.backward()

        def train_step():
            forward_backward()
            optimizer.step()

        for phase, fn in (
            ("forward", forward),
            ("forward_backward", forward_backward),
            ("train_step", train_step),
        ):
            try:
                cold_start = None
                if implementation == "compiled":
                    torch.cuda.synchronize(device)
                    start = time.perf_counter_ns()
                    fn()
                    torch.cuda.synchronize(device)
                    cold_start = (time.perf_counter_ns() - start) / 1e6
                for _ in range(warmup):
                    fn()
                torch.cuda.synchronize(device)
                samples = []
                for _ in range(steps):
                    start = time.perf_counter_ns()
                    fn()
                    torch.cuda.synchronize(device)
                    samples.append((time.perf_counter_ns() - start) / 1e6)
                rows.append(
                    {
                        "component": "small_model",
                        "shape": "B1_S512",
                        "implementation": implementation,
                        "phase": phase,
                        "compile_mode": "dynamic=False"
                        if implementation == "compiled"
                        else "eager",
                        "cold_start_ms": cold_start or "",
                        "samples_ms": json.dumps(samples),
                        "steady_p50_ms": sorted(samples)[len(samples) // 2],
                        "status": "pass",
                    }
                )
            except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                rows.append(
                    {
                        "component": "small_model",
                        "shape": "B1_S512",
                        "implementation": implementation,
                        "phase": phase,
                        "status": f"oom_or_runtime_error:{type(exc).__name__}",
                    }
                )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/compile_comparison.csv")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    if not torch.cuda.is_available():
        rows = [{"component": "attention", "status": "not_run_no_cuda"}]
    else:
        guard = configure_allocator_guard()
        device = torch.device("cuda")
        rows = _attention_rows(device, args.warmup, args.steps)
        if not args.skip_model:
            rows.extend(_model_rows(device, args.warmup, args.steps))
        for row in rows:
            row["allocator_guard_applied"] = guard["applied"]
            row["allocator_limit_mib"] = guard["limit_mib"]
            row["allocator_fraction"] = guard["fraction"]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
