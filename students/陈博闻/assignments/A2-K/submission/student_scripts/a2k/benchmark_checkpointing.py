from __future__ import annotations

import argparse
import math
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint_sequential

from cs336_basics.model import BasicsTransformerLM
from cs336_basics.nn_utils import cross_entropy
from cs336_basics.optimizer import AdamW
from student_scripts.a2k.common import cuda_event_bench, peak_memory, require_cuda, reset_peak_memory, write_csv


MEDIUM = {"d_model": 1024, "d_ff": 4096, "num_layers": 24, "num_heads": 16}


def build_model(context_length: int, device: torch.device) -> BasicsTransformerLM:
    torch.manual_seed(0)
    model = BasicsTransformerLM(vocab_size=10_000, context_length=context_length, rope_theta=10_000.0, **MEDIUM)
    return model.to(device)


def forward_with_checkpoint(model: BasicsTransformerLM, tokens: torch.Tensor, checkpoint_block_size: int | None):
    if checkpoint_block_size is None:
        return model(tokens)
    x = model.token_embeddings(tokens)
    segments = math.ceil(len(model.layers) / checkpoint_block_size)
    x = checkpoint_sequential(model.layers, segments, x, use_reentrant=False)
    x = model.ln_final(x)
    return model.lm_head(x)


def run_config(context_length: int, checkpoint_block_size: int | None, warmup_steps: int, measurement_steps: int, device: torch.device) -> dict:
    model = build_model(context_length, device)
    optimizer = AdamW(model.parameters(), lr=1e-3, weight_decay=0.01)
    torch.manual_seed(1)
    inputs = torch.randint(0, 10_000, (1, context_length), device=device)
    targets = torch.randint(0, 10_000, (1, context_length), device=device)

    def step():
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            logits = forward_with_checkpoint(model, inputs, checkpoint_block_size)
            loss = cross_entropy(logits, targets)
        loss.backward()
        optimizer.step()

    base = {
        "config_id": f"medium_ctx{context_length}_ckpt{checkpoint_block_size if checkpoint_block_size is not None else 'none'}",
        "model_size": "medium",
        "num_layers": MEDIUM["num_layers"],
        "context_length": context_length,
        "batch_size": 1,
        "dtype": "bf16_autocast_fp32_params",
        "checkpoint_block_size": "" if checkpoint_block_size is None else checkpoint_block_size,
        "nested": False,
        "warmup_steps": warmup_steps,
        "measurement_steps": measurement_steps,
    }
    try:
        reset_peak_memory()
        samples = cuda_event_bench(step, warmup_steps, measurement_steps)
        mem = peak_memory()
        return base | {
            "step_time_ms_samples": samples,
            "step_time_ms_p50": sorted(samples)[len(samples) // 2],
            "peak_allocated_mib": mem["peak_allocated_mib"],
            "peak_reserved_mib": mem["peak_reserved_mib"],
            "status": "ok",
        }
    except torch.OutOfMemoryError as exc:
        torch.cuda.empty_cache()
        return base | {
            "step_time_ms_samples": [],
            "step_time_ms_p50": "",
            "peak_allocated_mib": "",
            "peak_reserved_mib": "",
            "status": f"oom: {type(exc).__name__}",
        }
    finally:
        del model, optimizer, inputs, targets
        torch.cuda.empty_cache()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("local_results/a2k/checkpointing.csv"))
    parser.add_argument("--warmup-steps", type=int, default=3)
    parser.add_argument("--measurement-steps", type=int, default=5)
    args = parser.parse_args()
    device = require_cuda()
    rows = []
    standard_configs: list[int | None] = [None, 1, 2, 4, 8]
    for ckpt in standard_configs:
        rows.append(run_config(1024, ckpt, args.warmup_steps, args.measurement_steps, device))
    successful = [r for r in rows if r["status"] == "ok" and r["checkpoint_block_size"] != ""]
    best = min(successful, key=lambda r: float(r["peak_allocated_mib"])) if successful else None
    rows.append(run_config(2048, None, args.warmup_steps, args.measurement_steps, device))
    if best is not None:
        rows.append(run_config(2048, int(best["checkpoint_block_size"]), args.warmup_steps, args.measurement_steps, device))
    write_csv(args.output, rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
