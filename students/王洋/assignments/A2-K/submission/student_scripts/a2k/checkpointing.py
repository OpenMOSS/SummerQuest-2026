from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from torch.utils.checkpoint import checkpoint

from student_scripts.a2k.common import (
    append_csv,
    command_string,
    configure_formal_process,
    median,
    memory_peaks,
    public_environment,
    reset_peaks,
    write_json,
)

FIELDS = [
    "config_id",
    "model_size",
    "num_layers",
    "context_length",
    "batch_size",
    "dtype",
    "checkpoint_block_size",
    "nested",
    "warmup_steps",
    "measurement_steps",
    "step_time_ms_samples",
    "step_time_ms_p50",
    "peak_allocated_mib",
    "peak_reserved_mib",
    "status",
    "error_type",
    "command",
]


class CheckpointedTransformer(torch.nn.Module):
    def __init__(self, model: BasicsTransformerLM, block_size: int | None):
        super().__init__()
        self.model = model
        self.block_size = block_size

    def forward(self, token_ids: torch.Tensor) -> torch.Tensor:
        hidden = self.model.token_embeddings(token_ids)
        layers = self.model.layers
        if self.block_size is None:
            for layer in layers:
                hidden = layer(hidden)
        else:
            for start in range(0, len(layers), self.block_size):
                group = layers[start : start + self.block_size]

                def run_group(group_input: torch.Tensor, modules=group) -> torch.Tensor:
                    group_output = group_input
                    for module in modules:
                        group_output = module(group_output)
                    return group_output

                hidden = checkpoint(run_group, hidden, use_reentrant=False)
        return self.model.lm_head(self.model.ln_final(hidden))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure A2-K activation checkpointing.")
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--checkpoint-block-size", type=int, choices=[1, 2, 4, 8])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--metadata", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    fraction = configure_formal_process()
    environment = public_environment(fraction)
    torch.manual_seed(args.seed)
    config_id = f"medium_ctx{args.context_length}_block{args.checkpoint_block_size or 'none'}"
    row = {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": 24,
        "context_length": args.context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_parameters",
        "checkpoint_block_size": args.checkpoint_block_size or "none",
        "nested": False,
        "warmup_steps": args.warmup,
        "measurement_steps": args.steps,
        "status": "ok",
        "error_type": "",
        "command": command_string(),
    }
    try:
        base_model = BasicsTransformerLM(
            vocab_size=10_000,
            context_length=args.context_length,
            d_model=1024,
            d_ff=4096,
            num_layers=24,
            num_heads=16,
        ).cuda()
        model = CheckpointedTransformer(base_model, args.checkpoint_block_size)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        tokens = torch.randint(0, 10_000, (1, args.context_length + 1), device="cuda")
        inputs, targets = tokens[:, :-1], tokens[:, 1:]

        def train_step() -> float:
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                logits = model(inputs)
                loss = F.cross_entropy(logits.float().reshape(-1, logits.shape[-1]), targets.reshape(-1))
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        for _ in range(args.warmup):
            train_step()
            torch.cuda.synchronize()
        reset_peaks()
        samples = []
        for _ in range(args.steps):
            torch.cuda.synchronize()
            start = time.perf_counter()
            train_step()
            torch.cuda.synchronize()
            samples.append((time.perf_counter() - start) * 1000)
        peak_allocated, peak_reserved = memory_peaks()
        row.update(
            {
                "step_time_ms_samples": json.dumps(samples),
                "step_time_ms_p50": median(samples),
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
            }
        )
    except torch.cuda.OutOfMemoryError as exc:
        peak_allocated, peak_reserved = memory_peaks()
        row.update(
            {
                "status": "oom",
                "error_type": type(exc).__name__,
                "step_time_ms_samples": "[]",
                "step_time_ms_p50": "",
                "peak_allocated_mib": peak_allocated,
                "peak_reserved_mib": peak_reserved,
            }
        )
    append_csv(args.output, row, FIELDS)
    write_json(args.metadata, {"environment": environment, "seed": args.seed, "latest_row": row})
    print(row)


if __name__ == "__main__":
    main()
