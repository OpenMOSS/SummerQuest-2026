import argparse
import json
import shlex
import sys
from datetime import UTC, datetime
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW

from profiling.benchmark import (
    MODEL_CONFIGS,
    VOCAB_SIZE,
    build_model,
    collect_environment,
    run_step,
    synchronize,
)
from profiling.nvtx_ranges import attention_ranges


def run_profiled_train_step(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    with record_function("zero_grad"):
        optimizer.zero_grad(set_to_none=True)

    with record_function("forward"):
        logits = model(input_ids)

    with record_function("loss"):
        loss = cross_entropy(logits, targets)

    with record_function("backward"):
        loss.backward()

    with record_function("optimizer"):
        optimizer.step()


def capture_profile(
    model: BasicsTransformerLM,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    targets: torch.Tensor,
    warmup_steps: int,
    trace_path: Path,
) -> profile:
    for _ in range(warmup_steps - 1):
        run_step(model, optimizer, input_ids, targets, mode="train_step", precision="fp32")

    synchronize(input_ids.device)

    with attention_ranges():
        with profile(
            activities=[
                ProfilerActivity.CPU,
                ProfilerActivity.CUDA,
            ],
        ) as profiler:
            with record_function("profile/warmup"):
                run_step(model, optimizer, input_ids, targets, mode="train_step", precision="fp32")
                synchronize(input_ids.device)
            profiler.step()

            with record_function("profile/measure"):
                run_profiled_train_step(model, optimizer, input_ids, targets)
                synchronize(input_ids.device)
            profiler.step()

    profiler.export_chrome_trace(str(trace_path))
    return profiler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Capture one warmed-up Transformer train step.")
    parser.add_argument("--model-size", choices=MODEL_CONFIGS, default="small")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--trace-output", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path, required=True)
    parser.add_argument("--metadata-output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    device = torch.device("cuda")

    model = build_model(args.model_size, args.context_length, device)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate)
    input_ids = torch.randint(
        0,
        VOCAB_SIZE,
        (args.batch_size, args.context_length),
        device=device,
    )
    targets = torch.randint(
        0,
        VOCAB_SIZE,
        (args.batch_size, args.context_length),
        device=device,
    )

    for output in (args.trace_output, args.summary_output, args.metadata_output):
        output.parent.mkdir(parents=True, exist_ok=True)

    profiler = capture_profile(
        model=model,
        optimizer=optimizer,
        input_ids=input_ids,
        targets=targets,
        warmup_steps=args.warmup,
        trace_path=args.trace_output,
    )
    summary = profiler.key_averages().table(
        sort_by="self_cuda_time_total",
        row_limit=30,
    )
    args.summary_output.write_text(summary + "\n", encoding="utf-8")

    metadata = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "command": f"uv run python {shlex.join(sys.argv)}",
        "trace_output": args.trace_output.as_posix(),
        "summary_output": args.summary_output.as_posix(),
        "config": {
            "model_size": args.model_size,
            **MODEL_CONFIGS[args.model_size],
            "vocab_size": VOCAB_SIZE,
            "batch_size": args.batch_size,
            "context_length": args.context_length,
            "warmup_steps": args.warmup,
            "mode": "train_step",
            "measurement_steps": 1,
            "dtype": "fp32",
            "seed": args.seed,
            "learning_rate": args.learning_rate,
        },
        "environment": collect_environment(device),
    }
    args.metadata_output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(summary)
    print()
    print(json.dumps(metadata, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
