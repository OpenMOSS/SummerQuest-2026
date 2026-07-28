"""One-row eager/compiled attention or Stanford-small comparison."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

import torch
import torch.nn.functional as functional

from .common import (
    configure_formal_run,
    public_run_record,
    upsert_csv_rows,
    upsert_json_record,
)
from .attention_utils import (
    Phase,
    benchmark_attention_phase,
    make_attention_inputs,
    run_phase_once,
)

from cs336_basics.model import BasicsTransformerLM
from cs336_systems.a2k import explicit_attention
from cs336_systems.a2k.runtime import (
    benchmark_cuda,
    classify_exception,
    peak_memory_mib,
    timed_cold_start,
)


FIELDS = (
    "target",
    "model_size",
    "implementation",
    "seq_len",
    "head_dim",
    "batch_size",
    "dtype",
    "causal",
    "phase",
    "compile_mode",
    "fullgraph",
    "graph_break_count",
    "warmup_ms",
    "rep_ms",
    "cold_start_ms",
    "p20_ms",
    "p50_ms",
    "p80_ms",
    "sample_count",
    "measurement_elapsed_ms",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "allocator_limit_mib",
    "allocator_fraction",
    "free_memory_mib_at_start",
    "status",
    "error_type",
    "error",
)


SMALL_CONFIG = {
    "vocab_size": 10_000,
    "context_length": 512,
    "d_model": 768,
    "num_layers": 12,
    "num_heads": 12,
    "d_ff": 3072,
    "rope_theta": 10_000.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    parser.add_argument(
        "--target",
        choices=("attention", "small-model"),
        required=True,
    )
    parser.add_argument(
        "--implementation",
        choices=("eager", "compiled"),
        required=True,
    )
    parser.add_argument("--seq-len", type=int)
    parser.add_argument("--head-dim", type=int)
    parser.add_argument(
        "--phase",
        choices=(
            "forward",
            "backward",
            "forward-backward",
            "train-step",
        ),
        required=True,
    )
    parser.add_argument("--warmup-ms", type=float, default=100.0)
    parser.add_argument("--rep-ms", type=float, default=300.0)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()


def graph_break_count() -> int:
    try:
        counters = torch._dynamo.utils.counters["graph_break"]
    except (AttributeError, KeyError):
        return 0
    return int(sum(counters.values()))


def attention_row(
    args: argparse.Namespace,
    row: dict[str, Any],
) -> None:
    if args.seq_len is None or args.head_dim is None:
        raise ValueError("attention target requires --seq-len and --head-dim")
    if args.phase == "train-step":
        raise ValueError("attention target does not define train-step")
    phase: Phase = args.phase
    inputs = make_attention_inputs(
        sequence_length=args.seq_len,
        head_dim=args.head_dim,
        phase=phase,
        seed=args.seed,
    )

    def eager_forward() -> torch.Tensor:
        return explicit_attention(
            inputs.q,
            inputs.k,
            inputs.v,
            True,
        )

    forward: Callable[[], torch.Tensor] = eager_forward
    if args.implementation == "compiled":
        forward = torch.compile(
            eager_forward,
            fullgraph=True,
            mode="reduce-overhead",
        )
        row["cold_start_ms"] = timed_cold_start(
            lambda: run_phase_once(forward, inputs, phase)
        )
    inputs.clear_gradients()
    row.update(
        benchmark_attention_phase(
            forward,
            inputs,
            phase,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    )


def small_model_row(
    args: argparse.Namespace,
    row: dict[str, Any],
) -> None:
    if args.phase not in {
        "forward",
        "forward-backward",
        "train-step",
    }:
        raise ValueError(
            "small-model target requires forward, forward-backward, or train-step"
        )
    model = BasicsTransformerLM(**SMALL_CONFIG).to("cuda")
    tokens = torch.randint(
        0,
        SMALL_CONFIG["vocab_size"],
        (1, SMALL_CONFIG["context_length"]),
        device="cuda",
    )
    labels = torch.randint(
        0,
        SMALL_CONFIG["vocab_size"],
        (1, SMALL_CONFIG["context_length"]),
        device="cuda",
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    selected_model: Callable[[torch.Tensor], torch.Tensor] = model
    if args.implementation == "compiled":
        selected_model = torch.compile(
            model,
            fullgraph=False,
            mode="reduce-overhead",
        )

    def forward_loss() -> torch.Tensor:
        with torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            logits = selected_model(tokens)
        return functional.cross_entropy(
            logits.float().reshape(-1, SMALL_CONFIG["vocab_size"]),
            labels.reshape(-1),
        )

    def run_forward() -> None:
        with torch.no_grad(), torch.autocast(
            device_type="cuda",
            dtype=torch.bfloat16,
        ):
            selected_model(tokens)

    def run_forward_backward() -> None:
        optimizer.zero_grad(set_to_none=True)
        forward_loss().backward()

    def run_train_step() -> None:
        optimizer.zero_grad(set_to_none=True)
        forward_loss().backward()
        optimizer.step()

    measured = {
        "forward": run_forward,
        "forward-backward": run_forward_backward,
        "train-step": run_train_step,
    }[args.phase]
    if args.implementation == "compiled":
        row["cold_start_ms"] = timed_cold_start(measured)
    row.update(
        benchmark_cuda(
            measured,
            warmup_ms=args.warmup_ms,
            rep_ms=args.rep_ms,
        )
    )


def main() -> int:
    args = parse_args()
    run = configure_formal_run(seed=args.seed, tf32_enabled=False)
    seq_len = (
        args.seq_len
        if args.target == "attention"
        else SMALL_CONFIG["context_length"]
    )
    head_dim = args.head_dim if args.target == "attention" else 64
    config_id = (
        f"{args.target}-{args.implementation}-s{seq_len}-d{head_dim}-"
        f"{args.phase}"
    )
    row: dict[str, Any] = {
        "target": args.target,
        "model_size": (
            "attention-only" if args.target == "attention" else "small"
        ),
        "implementation": args.implementation,
        "seq_len": seq_len,
        "head_dim": head_dim,
        "batch_size": 1,
        "dtype": "bfloat16",
        "causal": True,
        "phase": args.phase,
        "compile_mode": (
            "reduce-overhead"
            if args.implementation == "compiled"
            else ""
        ),
        "fullgraph": (
            args.target == "attention"
            if args.implementation == "compiled"
            else ""
        ),
        "graph_break_count": "",
        "warmup_ms": args.warmup_ms,
        "rep_ms": args.rep_ms,
        "cold_start_ms": "",
        "allocator_limit_mib": run.allocator.allocator_limit_mib,
        "allocator_fraction": run.allocator.allocator_fraction,
        "free_memory_mib_at_start": run.free_memory_mib_at_start,
        "status": "ok",
        "error_type": "",
        "error": "",
    }
    try:
        if args.target == "attention":
            attention_row(args, row)
        else:
            small_model_row(args, row)
        row["graph_break_count"] = graph_break_count()
    except BaseException as error:
        row.update(classify_exception(error))
        row["graph_break_count"] = graph_break_count()
        try:
            row.update(peak_memory_mib())
        except RuntimeError:
            pass

    upsert_csv_rows(
        args.output,
        [row],
        key_fields=(
            "target",
            "implementation",
            "seq_len",
            "head_dim",
            "phase",
        ),
        fieldnames=FIELDS,
    )
    command = (
        "python -m student_scripts.a2k.compile_comparison "
        f"--target {args.target} --implementation {args.implementation} "
        f"--phase {args.phase} --warmup-ms {args.warmup_ms:g} "
        f"--rep-ms {args.rep_ms:g} --seed {args.seed}"
    )
    if args.target == "attention":
        command += (
            f" --seq-len {args.seq_len} --head-dim {args.head_dim}"
        )
    record = public_run_record(
        run=run,
        experiment="compile_comparison",
        command=command,
        timer="CUDA events; compilation recorded separately as wall time",
        warmup={"milliseconds": args.warmup_ms},
        measurement={
            "milliseconds": args.rep_ms,
            "quantiles": [0.2, 0.5, 0.8],
        },
        extra={
            "config_id": config_id,
            "status": row["status"],
            "compile_mode": row["compile_mode"],
            "fullgraph": row["fullgraph"],
            "graph_break_count": row["graph_break_count"],
        },
    )
    record["config_id"] = config_id
    upsert_json_record(
        args.metadata,
        record,
        key_fields=("experiment", "config_id"),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
