from __future__ import annotations

import argparse
import statistics
from collections.abc import Callable

import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from cs336_basics.model import BasicsTransformerLM
from student_scripts.a2k.common import add_common_args, cuda_event_time_ms, ensure_dirs, peak_memory_mib, require_cuda, reset_peak, set_allocator_limit, write_csv


MEDIUM_CONFIG = {
    "vocab_size": 10000,
    "context_length": 1024,
    "d_model": 1024,
    "num_layers": 24,
    "num_heads": 16,
    "d_ff": 4096,
    "rope_theta": 10000.0,
}


class CheckpointedLM(nn.Module):
    def __init__(self, base: BasicsTransformerLM, checkpoint_block_size: int | None):
        super().__init__()
        self.base = base
        self.checkpoint_block_size = checkpoint_block_size

    def _run_layers(self, x: torch.Tensor, start: int, end: int) -> torch.Tensor:
        for layer in self.base.layers[start:end]:
            x = layer(x)
        return x

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        x = self.base.token_embeddings(tokens)
        if self.checkpoint_block_size is None:
            for layer in self.base.layers:
                x = layer(x)
        else:
            block = self.checkpoint_block_size
            for start in range(0, len(self.base.layers), block):
                end = min(start + block, len(self.base.layers))
                x = checkpoint(lambda y, s=start, e=end: self._run_layers(y, s, e), x, use_reentrant=False)
        x = self.base.ln_final(x)
        return self.base.lm_head(x)


def make_step(context_length: int, checkpoint_block_size: int | None, device: torch.device, seed: int) -> Callable[[], None]:
    torch.manual_seed(seed)
    config = dict(MEDIUM_CONFIG)
    config["context_length"] = context_length
    model = CheckpointedLM(BasicsTransformerLM(**config), checkpoint_block_size).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    tokens = torch.randint(0, config["vocab_size"], (1, context_length), device=device)
    targets = torch.randint(0, config["vocab_size"], (1, context_length), device=device)
    loss_fn = nn.CrossEntropyLoss()

    def step() -> None:
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = model(tokens)
            loss = loss_fn(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1))
        loss.backward()
        optimizer.step()

    return step


def run_config(config_id: str, context_length: int, checkpoint_block_size: int | None, seed: int) -> dict:
    device = require_cuda()
    reset_peak()
    try:
        step = make_step(context_length, checkpoint_block_size, device, seed)
        samples = cuda_event_time_ms(step, warmup_steps=3, measurement_steps=5)
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "ok"
        p50 = statistics.median(samples)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        samples = []
        p50 = "NA"
        peak_allocated, peak_reserved = peak_memory_mib()
        status = "OOM"
    return {
        "config_id": config_id,
        "model_size": "medium",
        "num_layers": 24,
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "bf16",
        "checkpoint_block_size": checkpoint_block_size if checkpoint_block_size is not None else "none",
        "nested": False,
        "warmup_steps": 3,
        "measurement_steps": 5,
        "step_time_ms_samples": samples,
        "step_time_ms_p50": p50,
        "peak_allocated_mib": peak_allocated,
        "peak_reserved_mib": peak_reserved,
        "status": status,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    add_common_args(parser)
    args = parser.parse_args()
    ensure_dirs()
    set_allocator_limit()

    rows = []
    for block_size in [None, 1, 2, 4, 8]:
        label = "no_ckpt_ctx1024" if block_size is None else f"ckpt_b{block_size}_ctx1024"
        rows.append(run_config(label, 1024, block_size, args.seed))

    successful = [r for r in rows if r["status"] == "ok" and r["checkpoint_block_size"] != "none"]
    best = min(successful, key=lambda r: r["peak_allocated_mib"]) if successful else None
    rows.append(run_config("no_ckpt_ctx2048", 2048, None, args.seed))
    if best is not None:
        rows.append(run_config(f"ckpt_b{best['checkpoint_block_size']}_ctx2048", 2048, int(best["checkpoint_block_size"]), args.seed))

    write_csv(
        args.output_dir / "checkpointing.csv",
        rows,
        [
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
        ],
    )


if __name__ == "__main__":
    main()
