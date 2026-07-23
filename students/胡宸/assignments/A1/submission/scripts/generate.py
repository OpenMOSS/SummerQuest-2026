#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sample from a saved Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--vocab", required=True, type=Path)
    parser.add_argument("--merges", required=True, type=Path)
    parser.add_argument("--prompt", default="")
    parser.add_argument("--vocab-size", required=True, type=int)
    parser.add_argument("--context-length", type=int, default=256)
    parser.add_argument("--d-model", type=int, default=512)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--d-ff", type=int, default=1344)
    parser.add_argument("--rope-theta", type=float, default=10000.0)
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--norm-position", choices=["pre", "post"], default="pre")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", choices=["swiglu", "silu"], default="swiglu")
    parser.add_argument("--silu-d-ff", type=int)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    return parser.parse_args()


def sample_top_p(probs: torch.Tensor, p: float) -> int:
    if p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1))
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= p
    keep[0] = True
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum()
    return int(sorted_ids[torch.multinomial(filtered, num_samples=1)])


def main() -> int:
    args = parse_args()
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        use_rmsnorm=not args.no_rmsnorm,
        norm_position=args.norm_position,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
        silu_d_ff=args.silu_d_ff,
    ).to(args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    ids = tokenizer.encode(args.prompt)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx = torch.tensor([ids[-args.context_length :]], dtype=torch.long, device=args.device)
            logits = model(idx)[0, -1]
            probs = softmax(logits / args.temperature, dim=-1)
            ids.append(sample_top_p(probs, args.top_p))
    print(tokenizer.decode(ids))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
