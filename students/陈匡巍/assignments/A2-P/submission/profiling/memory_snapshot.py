"""Measure one memory configuration and retain its full snapshot only locally."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import torch.nn.functional as functional

from cs336_basics.model import BasicsTransformerLM
from profiling.common import (
    MIB,
    MODEL_CONFIGS,
    configure_gpu,
    gpu_metadata,
    write_json,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-size", choices=("large", "xl"), required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--mode", choices=("forward", "train_step"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2026)
    return parser.parse_args()


def largest_active_allocation() -> int:
    largest = 0
    for segment in torch.cuda.memory_snapshot():
        for block in segment.get("blocks", []):
            if str(block.get("state", "")).startswith("active"):
                largest = max(
                    largest,
                    int(block.get("requested_size", block.get("size", 0))),
                )
    return largest


def main() -> int:
    args = parse_args()
    allocator = configure_gpu()
    gpu = gpu_metadata()
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    timeline_samples: list[dict] = []
    failure_stage = "setup"
    history_enabled = False

    row = {
        "model_size": args.model_size,
        "batch_size": 1,
        "context_length": args.context_length,
        "mode": args.mode,
        "dtype": "fp32",
        "status": "error",
        "active_mib_after_step": "",
        "peak_allocated_mib": "",
        "reserved_mib_after_step": "",
        "peak_reserved_mib": "",
        "largest_active_allocation_mib": "",
        "error_type": "",
        "failure_stage": "",
    }
    try:
        model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            **MODEL_CONFIGS[args.model_size],
        ).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4) if args.mode == "train_step" else None
        tokens = torch.randint(0, 10_000, (1, args.context_length), device="cuda")
        targets = torch.randint(0, 10_000, (1, args.context_length), device="cuda")
        hooks = []

        def sample_layer(layer_index: int) -> None:
            timeline_samples.append(
                {
                    "stage": f"forward/layer_{layer_index:02d}",
                    "allocated_mib": torch.cuda.memory_allocated() / MIB,
                    "reserved_mib": torch.cuda.memory_reserved() / MIB,
                }
            )

        for layer_index, layer in enumerate(model.layers):
            hooks.append(layer.register_forward_hook(lambda _module, _inputs, _output, index=layer_index: sample_layer(index)))

        def run_step() -> None:
            if args.mode == "forward":
                with torch.no_grad():
                    model(tokens)
                return
            assert optimizer is not None
            optimizer.zero_grad(set_to_none=True)
            logits = model(tokens)
            loss = functional.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            timeline_samples.append(
                {
                    "stage": "backward/complete",
                    "allocated_mib": torch.cuda.memory_allocated() / MIB,
                    "reserved_mib": torch.cuda.memory_reserved() / MIB,
                }
            )
            optimizer.step()
            timeline_samples.append(
                {
                    "stage": "optimizer/complete",
                    "allocated_mib": torch.cuda.memory_allocated() / MIB,
                    "reserved_mib": torch.cuda.memory_reserved() / MIB,
                }
            )

        failure_stage = "warmup"
        timeline_samples.clear()
        run_step()
        torch.cuda.synchronize()
        timeline_samples.clear()

        failure_stage = "measurement"
        torch.cuda.memory._record_memory_history(max_entries=200_000)
        history_enabled = True
        torch.cuda.reset_peak_memory_stats()
        run_step()
        torch.cuda.synchronize()
        args.snapshot.parent.mkdir(parents=True, exist_ok=True)
        torch.cuda.memory._dump_snapshot(str(args.snapshot))
        active = torch.cuda.memory_allocated() / MIB
        reserved = torch.cuda.memory_reserved() / MIB
        peak_allocated = torch.cuda.max_memory_allocated() / MIB
        peak_reserved = torch.cuda.max_memory_reserved() / MIB
        largest = largest_active_allocation() / MIB
        row.update(
            {
                "status": "ok",
                "active_mib_after_step": active,
                "peak_allocated_mib": peak_allocated,
                "reserved_mib_after_step": reserved,
                "peak_reserved_mib": peak_reserved,
                "largest_active_allocation_mib": largest,
            }
        )
        for hook in hooks:
            hook.remove()
    except torch.OutOfMemoryError:
        row["status"] = "oom"
        row["error_type"] = "OutOfMemoryError"
        row["failure_stage"] = failure_stage
        row["active_mib_after_step"] = torch.cuda.memory_allocated() / MIB
        row["peak_allocated_mib"] = torch.cuda.max_memory_allocated() / MIB
        row["reserved_mib_after_step"] = torch.cuda.memory_reserved() / MIB
        row["peak_reserved_mib"] = torch.cuda.max_memory_reserved() / MIB
    except Exception as error:
        row["status"] = "error"
        row["error_type"] = type(error).__name__
        row["failure_stage"] = failure_stage
    finally:
        if history_enabled:
            torch.cuda.memory._record_memory_history(enabled=None)

    payload = {
        "row": row,
        "timeline_samples": timeline_samples,
        "snapshot_file": args.snapshot.name if row["status"] == "ok" else "",
        "snapshot_policy": "local only; never copied into the public submission",
        "allocator": allocator,
        "gpu": gpu,
        "command": (f"python -m profiling.memory_snapshot --model-size {args.model_size} --context-length {args.context_length} --mode {args.mode}"),
    }
    write_json(args.output, payload)
    print(json.dumps(payload["row"], ensure_ascii=False))
    return 0 if row["status"] in {"ok", "oom"} else 1


if __name__ == "__main__":
    raise SystemExit(main())
