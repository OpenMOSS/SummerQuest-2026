"""Cold-start and steady-state torch.compile comparisons."""

from __future__ import annotations

import argparse
import gc
import os
import subprocess
import sys
import tempfile
import time
from collections.abc import Callable
from pathlib import Path

import torch
import torch.nn.functional as functional
import triton

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k.attention import explicit_attention
from student_scripts.a2k.common import (
    MODEL_CONFIGS,
    configure_single_gpu,
    peak_memory,
    public_gpu_metadata,
    read_json,
    write_csv,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path)
    parser.add_argument("--metadata-output", type=Path)
    parser.add_argument(
        "--single-kind",
        choices=("attention", "small_model"),
    )
    parser.add_argument("--sequence-length", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--workload",
        choices=("forward", "forward_backward", "full_training_step"),
    )
    parser.add_argument("--single-output", type=Path)
    return parser.parse_args()


def do_bench(action: Callable) -> tuple[float, float, float]:
    values = triton.testing.do_bench(action, warmup=100, rep=300, quantiles=[0.2, 0.5, 0.8])
    return tuple(float(value) for value in values)


def attention_rows(sequence_length: int, head_dim: int) -> list[dict]:
    rows = []
    torch.manual_seed(2026)
    q = torch.randn(
        1,
        sequence_length,
        head_dim,
        device="cuda",
        dtype=torch.bfloat16,
        requires_grad=True,
    )
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    grad_output = torch.randn_like(q)
    for implementation in ("eager", "compiled"):
        function = explicit_attention if implementation == "eager" else torch.compile(explicit_attention, backend="inductor", fullgraph=True)

        def action() -> None:
            q.grad = None
            k.grad = None
            v.grad = None
            output = function(q, k, v, True)
            output.backward(grad_output)

        cold_compile_ms: float | str = ""
        try:
            torch.cuda.synchronize()
            cold_start = time.perf_counter()
            action()
            torch.cuda.synchronize()
            if implementation == "compiled":
                cold_compile_ms = (time.perf_counter() - cold_start) * 1000
            p20, p50, p80 = do_bench(action)
            torch.cuda.reset_peak_memory_stats()
            action()
            torch.cuda.synchronize()
            allocated, reserved = peak_memory()
            status = "ok"
            error_type = ""
        except torch.OutOfMemoryError:
            p20 = p50 = p80 = allocated = reserved = ""
            status = "oom"
            error_type = "OutOfMemoryError"
            torch.cuda.empty_cache()
        except Exception as error:
            p20 = p50 = p80 = allocated = reserved = ""
            status = "error"
            error_type = type(error).__name__
            torch.cuda.empty_cache()
        rows.append(
            {
                "kind": "attention",
                "workload": "forward_backward",
                "sequence_length": sequence_length,
                "head_dim": head_dim,
                "implementation": implementation,
                "cold_compile_ms": cold_compile_ms,
                "cold_start_context": ("fresh_process_and_empty_inductor_cache" if implementation == "compiled" else ""),
                "steady_p20_ms": p20,
                "steady_p50_ms": p50,
                "steady_p80_ms": p80,
                "peak_allocated_mib": allocated,
                "peak_reserved_mib": reserved,
                "status": status,
                "error_type": error_type,
            }
        )
    gc.collect()
    torch.cuda.empty_cache()
    return rows


def small_model_rows(workloads: tuple[str, ...]) -> list[dict]:
    rows: list[dict] = []
    torch.manual_seed(2026)
    model = BasicsTransformerLM(
        vocab_size=10_000,
        context_length=512,
        **MODEL_CONFIGS["small"],
    ).cuda()
    compiled_model = torch.compile(model, backend="inductor", fullgraph=True)
    tokens = torch.randint(0, 10_000, (1, 512), device="cuda")
    targets = torch.randint(0, 10_000, (1, 512), device="cuda")

    for workload in workloads:
        for implementation in ("eager", "compiled"):
            selected_model = model if implementation == "eager" else compiled_model
            optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

            def action() -> torch.Tensor | None:
                if workload == "forward":
                    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
                        return selected_model(tokens)
                optimizer.zero_grad(set_to_none=True)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = selected_model(tokens)
                    loss = functional.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]),
                        targets.reshape(-1),
                    )
                loss.backward()
                if workload == "full_training_step":
                    optimizer.step()
                return None

            cold_compile_ms: float | str = ""
            try:
                torch.cuda.synchronize()
                cold_start = time.perf_counter()
                action()
                torch.cuda.synchronize()
                if implementation == "compiled":
                    cold_compile_ms = (time.perf_counter() - cold_start) * 1000
                p20, p50, p80 = do_bench(action)
                torch.cuda.reset_peak_memory_stats()
                action()
                torch.cuda.synchronize()
                allocated, reserved = peak_memory()
                status = "ok"
                error_type = ""
            except torch.OutOfMemoryError:
                p20 = p50 = p80 = allocated = reserved = ""
                status = "oom"
                error_type = "OutOfMemoryError"
                torch.cuda.empty_cache()
            except Exception as error:
                p20 = p50 = p80 = allocated = reserved = ""
                status = "error"
                error_type = type(error).__name__
                torch.cuda.empty_cache()
            rows.append(
                {
                    "kind": "small_model",
                    "workload": workload,
                    "sequence_length": 512,
                    "head_dim": "",
                    "implementation": implementation,
                    "cold_compile_ms": cold_compile_ms,
                    "cold_start_context": ("fresh_process_and_empty_inductor_cache" if implementation == "compiled" else ""),
                    "steady_p20_ms": p20,
                    "steady_p50_ms": p50,
                    "steady_p80_ms": p80,
                    "peak_allocated_mib": allocated,
                    "peak_reserved_mib": reserved,
                    "status": status,
                    "error_type": error_type,
                }
            )
    return rows


def isolated_run(
    single_arguments: list[str],
    cache_directory: Path,
    output: Path,
) -> dict:
    command = [
        sys.executable,
        "-m",
        "student_scripts.a2k.run_compile_comparison",
        *single_arguments,
        "--single-output",
        str(output),
    ]
    environment = os.environ.copy()
    environment.setdefault("CUDA_VISIBLE_DEVICES", "0")
    environment["TORCHINDUCTOR_CACHE_DIR"] = str(cache_directory)
    subprocess.run(command, env=environment, check=True)
    return read_json(output)


def main() -> int:
    args = parse_args()
    if args.single_kind is not None:
        if args.single_output is None:
            raise ValueError("--single-output is required for an isolated run")
        allocator = configure_single_gpu()
        gpu = public_gpu_metadata()
        torch.backends.cuda.matmul.allow_tf32 = True
        if args.single_kind == "attention":
            if args.sequence_length is None or args.head_dim is None:
                raise ValueError("attention run requires sequence length and head dim")
            rows = attention_rows(args.sequence_length, args.head_dim)
        else:
            if args.workload is None:
                raise ValueError("small-model run requires --workload")
            rows = small_model_rows((args.workload,))
        write_json(
            args.single_output,
            {
                "rows": rows,
                "allocator": allocator,
                "gpu": gpu,
            },
        )
        return 0

    if args.output is None:
        raise ValueError("--output is required for the full matrix")
    rows: list[dict] = []
    runs: list[dict] = []
    with tempfile.TemporaryDirectory(prefix="a2k-compile-") as temporary:
        scratch = Path(temporary)
        for index, (sequence_length, head_dim) in enumerate(((512, 64), (2048, 128), (8192, 128))):
            payload = isolated_run(
                [
                    "--single-kind",
                    "attention",
                    "--sequence-length",
                    str(sequence_length),
                    "--head-dim",
                    str(head_dim),
                ],
                scratch / f"cache-attention-{index}",
                scratch / f"attention-{index}.json",
            )
            rows.extend(payload["rows"])
            runs.append(
                {
                    "kind": "attention",
                    "sequence_length": sequence_length,
                    "head_dim": head_dim,
                    "allocator": payload["allocator"],
                    "gpu": payload["gpu"],
                }
            )
        for index, workload in enumerate(("forward", "forward_backward", "full_training_step")):
            payload = isolated_run(
                [
                    "--single-kind",
                    "small_model",
                    "--workload",
                    workload,
                ],
                scratch / f"cache-small-{index}",
                scratch / f"small-{index}.json",
            )
            rows.extend(payload["rows"])
            runs.append(
                {
                    "kind": "small_model",
                    "workload": workload,
                    "allocator": payload["allocator"],
                    "gpu": payload["gpu"],
                }
            )

    write_csv(args.output, rows)
    if args.metadata_output:
        write_json(
            args.metadata_output,
            {
                "experiment": "torch_compile",
                "process_isolation": ("one fresh process and empty Inductor cache per workload"),
                "runs": runs,
                "backend": "inductor",
                "cold_start_separated": True,
                "steady_timer": "triton.testing.do_bench",
                "warmup_ms": 100,
                "rep_ms": 300,
                "quantiles": [0.2, 0.5, 0.8],
                "command": ("python -m student_scripts.a2k.run_compile_comparison --output results/compile_comparison.csv"),
            },
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
