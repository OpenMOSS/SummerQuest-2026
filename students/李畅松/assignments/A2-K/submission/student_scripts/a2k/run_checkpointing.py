from __future__ import annotations

import argparse

import torch
import torch.nn.functional as F
from cs336_basics.model import BasicsTransformerLM
from torch.utils.checkpoint import checkpoint

from profiling.benchmark import MODEL_CONFIGS
from student_scripts.a2k.common import DTYPES, environment, failure_record, memory_stats, percentiles, write_json


def forward_checkpointed(model, tokens, block_size):
    x = model.token_embeddings(tokens)
    if block_size <= 0:
        for layer in model.layers:
            x = layer(x)
    else:
        for start in range(0, len(model.layers), block_size):
            layers = model.layers[start : start + block_size]
            def run(value, layers=layers):
                for layer in layers:
                    value = layer(value)
                return value
            x = checkpoint(run, x, use_reentrant=False)
    return model.lm_head(model.ln_final(x))


def main(args):
    stage = "setup"
    try:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU is required")
        config = dict(MODEL_CONFIGS[args.model_size])
        config["num_layers"] = args.num_layers
        model = BasicsTransformerLM(args.vocab_size, args.context_length, **config).cuda()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
        tokens = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        targets = torch.randint(args.vocab_size, (args.batch_size, args.context_length), device="cuda")
        amp_dtype = DTYPES[args.dtype]

        def step():
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=amp_dtype, enabled=amp_dtype != torch.float32):
                logits = forward_checkpointed(model, tokens, args.checkpoint_block_size)
                loss = F.cross_entropy(logits.flatten(0, 1).float(), targets.flatten())
            loss.backward()
            optimizer.step()
            return loss

        stage = "warmup"
        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        stage = "measure"
        timings = []
        for _ in range(args.steps):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            loss = step()
            end.record()
            end.synchronize()
            timings.append(start.elapsed_time(end))
        result = {
            "status": "ok",
            "config": vars(args),
            "model_config": config,
            "environment": environment(),
            "timing": percentiles(timings),
            "last_loss": loss.item(),
        }
        result.update(memory_stats())
        write_json(args.output, result)
    except Exception as exc:  # noqa: BLE001 - OOM and runtime failures must be serialized.
        write_json(args.output, failure_record(args, exc, stage))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--model-size", choices=MODEL_CONFIGS, default="medium")
    p.add_argument("--num-layers", type=int, default=24)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--context-length", type=int, required=True)
    p.add_argument("--vocab-size", type=int, default=10_000)
    p.add_argument("--dtype", choices=DTYPES, default="bf16")
    p.add_argument("--checkpoint-block-size", type=int, default=0)
    p.add_argument("--warmup", type=int, default=3)
    p.add_argument("--steps", type=int, default=5)
    p.add_argument("--output", required=True)
    main(p.parse_args())
