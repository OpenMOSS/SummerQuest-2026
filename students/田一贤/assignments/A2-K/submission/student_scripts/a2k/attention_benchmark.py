from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from .common import configure_allocator_guard, measure_attention_phase, native_attention


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/attention_baseline.csv")
    )
    parser.add_argument(
        "--device", default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    args = parser.parse_args()
    guard = configure_allocator_guard() if args.device.startswith("cuda") else {"applied": False}
    device = torch.device(args.device)
    rows = []
    for seq in (512, 2048, 8192):
        for dim in (64, 128):
            for phase in ("forward", "backward", "forward_backward"):
                if device.type != "cuda":
                    rows.append(
                        {
                            "sequence": seq,
                            "head_dim": dim,
                            "dtype": "bfloat16",
                            "phase": phase,
                            "allocator_guard_applied": guard["applied"],
                            "allocator_limit_mib": guard.get("limit_mib", 23552),
                            "allocator_fraction": guard.get("fraction", 0.0),
                            "status": "not_run_no_cuda",
                        }
                    )
                    continue
                q = torch.randn(
                    1, seq, dim, device=device, dtype=torch.bfloat16, requires_grad=True
                )
                k = torch.randn_like(q, requires_grad=True)
                v = torch.randn_like(q, requires_grad=True)
                try:
                    measured = measure_attention_phase(
                        lambda a, b, c: native_attention(a, b, c, True),
                        q,
                        k,
                        v,
                        phase,
                        warmup=args.warmup,
                        steps=args.steps,
                    )
                    rows.append(
                        {
                            "sequence": seq,
                            "head_dim": dim,
                            "dtype": "bfloat16",
                            "phase": phase,
                            "allocator_guard_applied": guard["applied"],
                            "allocator_limit_mib": guard.get("limit_mib", 23552),
                            "allocator_fraction": guard.get("fraction", 0.0),
                            **measured,
                            "status": "pass",
                        }
                    )
                except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                    rows.append(
                        {
                            "sequence": seq,
                            "head_dim": dim,
                            "dtype": "bfloat16",
                            "phase": phase,
                            "status": f"oom_or_runtime_error:{type(exc).__name__}",
                        }
                    )
                    if device.type == "cuda":
                        torch.cuda.empty_cache()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            if isinstance(row.get("samples_ms"), list):
                row["samples_ms"] = json.dumps(row["samples_ms"])
            writer.writerow(row)


if __name__ == "__main__":
    main()
