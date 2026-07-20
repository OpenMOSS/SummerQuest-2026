"""Run one full training step to test whether a batch size fits on CUDA."""

from __future__ import annotations

import argparse

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.nn_utils import cross_entropy


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("batch_size", type=int)
    args = parser.parse_args()
    model = TransformerLM(10_000, 256, 512, 4, 16, 1344, device="cuda")
    inputs = torch.randint(0, 10_000, (args.batch_size, 256), device="cuda")
    targets = torch.randint(0, 10_000, (args.batch_size, 256), device="cuda")
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        loss = cross_entropy(model(inputs), targets)
    loss.backward()
    torch.cuda.synchronize()
    print(f"batch_size={args.batch_size} loss={float(loss.detach()):.4f} fits")


if __name__ == "__main__":
    main()
