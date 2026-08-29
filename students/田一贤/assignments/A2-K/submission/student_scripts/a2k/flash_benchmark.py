from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from cs336_systems.a2k.attention import FlashAttentionTriton

from .common import configure_allocator_guard, measure_attention_phase, native_attention


def _write_rows(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for source in rows:
            row = dict(source)
            if isinstance(row.get("samples_ms"), list):
                row["samples_ms"] = json.dumps(row["samples_ms"])
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output", type=Path, default=Path("results/flash_benchmark.csv")
    )
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--include-boundary", action="store_true")
    parser.add_argument(
        "--boundary-only",
        action="store_true",
        help="Measure only the optional sequence-length 16384 boundary.",
    )
    parser.add_argument(
        "--skip-compiled-boundary",
        action="store_true",
        help="Skip the optional compiled implementation at sequence length 16384.",
    )
    args = parser.parse_args()
    if args.boundary_only:
        sequences = (16384,)
    elif args.include_boundary:
        sequences = (512, 2048, 8192, 16384)
    else:
        sequences = (512, 2048, 8192)
    rows = []
    if not torch.cuda.is_available():
        rows.append({"status": "not_run_no_cuda"})
    else:
        guard = configure_allocator_guard()
        device = torch.device("cuda")
        compiled = torch.compile(
            lambda q, k, v: native_attention(q, k, v, True), dynamic=False
        )
        implementations = {
            "eager": lambda q, k, v: native_attention(q, k, v, True),
            "compiled": compiled,
            "triton": lambda q, k, v: FlashAttentionTriton.apply(q, k, v, True),
        }
        for seq in sequences:
            for dim in (64, 128):
                for name, implementation in implementations.items():
                    if (
                        args.skip_compiled_boundary
                        and seq == 16384
                        and name == "compiled"
                    ):
                        continue
                    for phase in ("forward", "backward", "forward_backward"):
                        base = {
                            "implementation": name,
                            "sequence": seq,
                            "head_dim": dim,
                            "dtype": "bfloat16",
                            "causal": True,
                            "phase": phase,
                            "q_tile": 64 if name == "triton" else "",
                            "k_tile": 64 if name == "triton" else "",
                            "num_warps": 4 if name == "triton" else "",
                            "num_stages": 2 if name == "triton" else "",
                            "allocator_guard_applied": guard["applied"],
                            "allocator_limit_mib": guard["limit_mib"],
                            "allocator_fraction": guard["fraction"],
                        }
                        q = torch.randn(
                            1,
                            seq,
                            dim,
                            device=device,
                            dtype=torch.bfloat16,
                            requires_grad=True,
                        )
                        k = torch.randn_like(q, requires_grad=True)
                        v = torch.randn_like(q, requires_grad=True)
                        try:
                            measured = measure_attention_phase(
                                implementation,
                                q,
                                k,
                                v,
                                phase,
                                warmup=args.warmup,
                                steps=args.steps,
                            )
                            rows.append({**base, **measured, "status": "pass"})
                        except (RuntimeError, torch.cuda.OutOfMemoryError) as exc:
                            rows.append(
                                {
                                    **base,
                                    "status": f"oom_or_runtime_error:{type(exc).__name__}",
                                }
                            )
                        finally:
                            torch.cuda.empty_cache()
                        _write_rows(args.output, rows)

        eager = {
            (row["sequence"], row["head_dim"], row["phase"]): row["p50_ms"]
            for row in rows
            if row.get("implementation") == "eager" and row.get("status") == "pass"
        }
        for row in rows:
            key = (row.get("sequence"), row.get("head_dim"), row.get("phase"))
            if row.get("status") == "pass" and key in eager:
                row["speedup_vs_eager"] = eager[key] / row["p50_ms"]
    _write_rows(args.output, rows)


if __name__ == "__main__":
    main()
