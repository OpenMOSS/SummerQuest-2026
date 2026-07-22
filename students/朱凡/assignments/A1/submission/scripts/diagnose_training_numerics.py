"""Diagnose finite losses/gradients for the full TinyStories training shape."""

from __future__ import annotations

import argparse

import numpy as np
import torch

from cs336_basics.data import get_batch
from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import clip_gradients, cross_entropy
from cs336_basics.optimizer import AdamW


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--compile", action="store_true")
    args = parser.parse_args()

    torch.manual_seed(1337)
    np.random.seed(1337)
    data = np.memmap("data/ts_train.bin", dtype=np.uint16, mode="r")
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344, device="cuda")
    optimizer = AdamW(model.parameters(), lr=1e-4, betas=(0.9, 0.95), eps=1e-8, weight_decay=0.1)
    training_model = torch.compile(model) if args.compile else model

    for step in range(1, args.steps + 1):
        inputs, targets = get_batch(data, args.batch_size, 256, "cuda")
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            loss = cross_entropy(training_model(inputs), targets)
        loss.backward()
        gradients_finite = all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all()) for parameter in model.parameters()
        )
        clip_gradients(model.parameters(), 1.0)
        optimizer.step()
        parameters_finite = all(bool(torch.isfinite(parameter).all()) for parameter in model.parameters())
        print(
            f"step={step} loss={float(loss.detach())} gradients_finite={gradients_finite} "
            f"parameters_finite={parameters_finite}",
            flush=True,
        )
        if not gradients_finite or not parameters_finite or not bool(torch.isfinite(loss)):
            raise RuntimeError("non-finite training state")


if __name__ == "__main__":
    main()
