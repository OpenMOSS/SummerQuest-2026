from __future__ import annotations

import sys

import torch

from cs336_systems.a2k.attention import FlashAttentionTriton


def main() -> int:
    dim = int(sys.argv[1]) if len(sys.argv) > 1 else 32
    seq = int(sys.argv[2]) if len(sys.argv) > 2 else 16
    causal = bool(int(sys.argv[3])) if len(sys.argv) > 3 else False
    print("start", dim, seq, causal, flush=True)
    q = torch.randn((1, seq, dim), device="cuda", dtype=torch.float32, requires_grad=True)
    k = torch.randn_like(q, requires_grad=True)
    v = torch.randn_like(q, requires_grad=True)
    print("inputs", flush=True)
    out = FlashAttentionTriton.apply(q, k, v, causal)
    torch.cuda.synchronize()
    print("forward", float(out.abs().max()), flush=True)
    out.sum().backward()
    torch.cuda.synchronize()
    print("backward", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
