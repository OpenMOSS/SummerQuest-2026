from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

ROOT = Path(__file__).resolve().parents[2]
CS336_BASICS = ROOT / "cs336-basics"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(CS336_BASICS) not in sys.path:
    sys.path.insert(0, str(CS336_BASICS))

from cs336_basics.model import BasicsTransformerLM
from student_scripts.a2k.common import bench, memory_stats, set_allocator_limit, set_seed, summarize, write_csv


CONFIGS = {
    "medium": {"vocab_size": 10000, "d_model": 1024, "num_layers": 24, "num_heads": 16, "d_ff": 4096},
}


class CheckpointedLM(torch.nn.Module):
    def __init__(self, model: BasicsTransformerLM, block_size: int | None):
        super().__init__()
        self.model = model
        self.block_size = block_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        hidden = self.model.token_embeddings(x)
        layers = list(self.model.layers)
        if self.block_size is None:
            for layer in layers:
                hidden = layer(hidden)
        else:
            for start in range(0, len(layers), self.block_size):
                chunk = torch.nn.Sequential(*layers[start : start + self.block_size])
                hidden = checkpoint(chunk, hidden, use_reentrant=False)
        hidden = self.model.ln_final(hidden)
        return self.model.lm_head(hidden)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--context-length", type=int, action="append")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument("--block-size", action="append", default=None)
    parser.add_argument("--allocator-limit-mib", type=int, default=23552)
    return parser.parse_args()


def run_one(context_length: int, block_size: int | None, warmup: int, steps: int) -> dict[str, object]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(2026 + context_length + (block_size or 0))
    cfg = CONFIGS["medium"]
    base = BasicsTransformerLM(context_length=context_length, rope_theta=10000.0, **cfg).to(device)
    model = CheckpointedLM(base, block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, cfg["vocab_size"], (1, context_length + 1), device=device)
    inp, labels = x[:, :-1], x[:, 1:]

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"):
            logits = model(inp)
            loss = F.cross_entropy(logits.flatten(0, -2).float(), labels.flatten())
        loss.backward()
        optimizer.step()

    samples, status, error = bench(step, warmup, steps)
    return {
        "config_id": f"medium_ctx{context_length}_ckpt{block_size if block_size is not None else 'none'}",
        "model_size": "medium",
        "num_layers": cfg["num_layers"],
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast",
        "checkpoint_block_size": "" if block_size is None else block_size,
        "nested": False,
        "warmup_steps": warmup,
        "measurement_steps": steps,
        "step_time_ms_samples": json.dumps(samples),
        "step_time_ms_p50": summarize(samples)["p50_ms"],
        **memory_stats(),
        "status": status,
        "error": error,
    }


def main() -> int:
    args = parse_args()
    set_allocator_limit(args.allocator_limit_mib)
    contexts = args.context_length or [1024]
    blocks = []
    if args.block_size is None:
        blocks = [None, 1, 2, 4, 8]
    else:
        for value in args.block_size:
            blocks.append(None if value == "none" else int(value))
    rows = []
    for ctx in contexts:
        for block in blocks:
            rows.append(run_one(ctx, block, args.warmup, args.steps))
    write_csv(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
