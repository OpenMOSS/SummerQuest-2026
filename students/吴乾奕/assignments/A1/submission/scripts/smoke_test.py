#!/usr/bin/env python3
"""Run a quick CPU forward/backward/checkpoint/generation integration test."""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

import numpy as np
import torch

from cs336_basics.checkpoint import load_checkpoint, save_checkpoint
from cs336_basics.data import get_batch
from cs336_basics.generation import generate_token_ids
from cs336_basics.losses import cross_entropy
from cs336_basics.optim import AdamW, gradient_clipping
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(7)
    np.random.seed(7)
    vocab_size = 32
    context_length = 16
    repeated = np.tile(np.arange(vocab_size, dtype=np.uint16), 256)
    model = TransformerLM(
        vocab_size=vocab_size,
        context_length=context_length,
        d_model=32,
        num_layers=2,
        num_heads=4,
        d_ff=64,
        rope_theta=10_000,
    )
    optimizer = AdamW(model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
    losses: list[float] = []
    for _ in range(args.steps):
        inputs, targets = get_batch(repeated, batch_size=8, context_length=context_length, device="cpu")
        optimizer.zero_grad(set_to_none=True)
        loss = cross_entropy(model(inputs), targets)
        loss.backward()
        gradient_clipping(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach()))

    with tempfile.TemporaryDirectory() as temporary_directory:
        checkpoint_path = Path(temporary_directory) / "checkpoint.pt"
        save_checkpoint(model, optimizer, args.steps, checkpoint_path)
        restored_model = TransformerLM(
            vocab_size=vocab_size,
            context_length=context_length,
            d_model=32,
            num_layers=2,
            num_heads=4,
            d_ff=64,
            rope_theta=10_000,
        )
        restored_optimizer = AdamW(restored_model.parameters(), lr=3e-3, betas=(0.9, 0.95), weight_decay=0.01)
        restored_iteration = load_checkpoint(checkpoint_path, restored_model, restored_optimizer)
        for original, restored in zip(model.parameters(), restored_model.parameters(), strict=True):
            torch.testing.assert_close(original, restored)

    generated = generate_token_ids(
        restored_model,
        [0, 1, 2],
        max_new_tokens=8,
        context_length=context_length,
        device="cpu",
        temperature=0,
    )
    result = {
        "steps": args.steps,
        "initial_loss": losses[0],
        "final_loss": losses[-1],
        "loss_decreased": losses[-1] < losses[0],
        "restored_iteration": restored_iteration,
        "generated_ids": generated,
    }
    if not result["loss_decreased"]:
        raise AssertionError(f"smoke-test loss did not decrease: {losses[0]} -> {losses[-1]}")
    rendered = json.dumps(result, indent=2, sort_keys=True)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
