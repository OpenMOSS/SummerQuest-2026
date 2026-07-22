"""Load a trained checkpoint + tokenizer and sample text (temperature / top-p).

Example:
    python scripts/generate_text.py \
        --ckpt artifacts/tinystories_main.pt \
        --tokenizer artifacts/tinystories_tokenizer.pkl \
        --prompt "Once upon a time" --temperature 0.8 --top-p 0.95 --max-new-tokens 256
"""

from __future__ import annotations

import argparse
import pickle

import torch

from cs336_basics.generate import generate
from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def load_tokenizer(path: str) -> Tokenizer:
    with open(path, "rb") as f:
        data = pickle.load(f)
    return Tokenizer(data["vocab"], data["merges"], data["special_tokens"])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--prompt", default="")
    p.add_argument("--max-new-tokens", type=int, default=256)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top-p", type=float, default=0.95)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda")
    # model shape (must match the checkpoint)
    p.add_argument("--vocab-size", type=int, default=10000)
    p.add_argument("--context-length", type=int, default=256)
    p.add_argument("--d-model", type=int, default=512)
    p.add_argument("--d-ff", type=int, default=1344)
    p.add_argument("--num-layers", type=int, default=4)
    p.add_argument("--num-heads", type=int, default=16)
    p.add_argument("--rope-theta", type=float, default=10000.0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    device = args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu"

    tokenizer = load_tokenizer(args.tokenizer)
    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
        device=device,
    ).to(device)

    state = torch.load(args.ckpt, map_location=device)
    model.load_state_dict(state["model"] if "model" in state else state)

    text = generate(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print("=== PROMPT ===")
    print(args.prompt)
    print("=== SAMPLE ===")
    print(text)


if __name__ == "__main__":
    main()
