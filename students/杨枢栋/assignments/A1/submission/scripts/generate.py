from __future__ import annotations

import argparse
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import AdamW
from cs336_basics.training import load_checkpoint
from scripts.train_lm import sample_text


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--norm-mode", default="pre")
    parser.add_argument("--no-rmsnorm", action="store_true")
    parser.add_argument("--no-rope", action="store_true")
    parser.add_argument("--ffn-type", default="swiglu")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with open(args.tokenizer, "rb") as file:
        tokenizer_state = pickle.load(file)
    tokenizer = Tokenizer(
        tokenizer_state["vocab"],
        tokenizer_state["merges"],
        tokenizer_state.get("special_tokens", ["<|endoftext|>"]),
    )
    model = TransformerLM(
        args.vocab_size,
        args.context_length,
        args.d_model,
        args.num_layers,
        args.num_heads,
        args.d_ff,
        10000.0,
        norm_mode=args.norm_mode,
        use_rmsnorm=not args.no_rmsnorm,
        use_rope=not args.no_rope,
        ffn_type=args.ffn_type,
    ).to(args.device)
    optimizer = AdamW(model.parameters(), lr=0.0)
    load_checkpoint(args.checkpoint, model, optimizer)
    print(
        sample_text(
            model,
            tokenizer,
            args.prompt,
            args.max_new_tokens,
            args.temperature,
            args.top_p,
            args.device,
        )
    )


if __name__ == "__main__":
    main()
