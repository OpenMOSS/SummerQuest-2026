import torch.cuda.nvtx as nvtx
import os

_enable = os.environ.get("ENABLE_NVTX", "0") == "1"

def nvtx_range(name):
    if _enable:
        return nvtx.range(name)
    else:
        from contextlib import nullcontext
        return nullcontext()


