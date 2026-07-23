from __future__ import annotations

from contextlib import contextmanager, nullcontext
from typing import Iterator

import torch


@contextmanager
def nvtx_range(name: str, enabled: bool = True) -> Iterator[None]:
    """Push an NVTX range when CUDA/NVTX is available.

    Keeping this as a tiny wrapper lets the benchmark run on CPU-only machines
    and on local GPUs without changing the measurement code.
    """

    if not enabled or not torch.cuda.is_available():
        with nullcontext():
            yield
        return

    torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        torch.cuda.nvtx.range_pop()


def synchronize_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
