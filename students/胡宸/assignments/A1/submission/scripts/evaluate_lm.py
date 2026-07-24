#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from cs336_basics.model import TransformerLM, cross_entropy, softmax
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import get_batch


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a saved Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--eval-data", required=True, type=Path)
    parser.add_argument("--dtype", default="uint16", choices=["uint16", "uint32"])
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
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-batches", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-threads", type=int)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--log-file", type=Path)
    parser.add_argument("--vocab", type=Path)
    parser.add_argument("--merges", type=Path)
    parser.add_argument("--special-token", action="append", default=["<|endoftext|>"])
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    return parser.parse_args()


def sample_top_p(probs: torch.Tensor, p: float) -> int:
    if p >= 1.0:
        return int(torch.multinomial(probs, num_samples=1).item())
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    keep = cumulative <= p
    keep[0] = True
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum()
    sampled = torch.multinomial(filtered, num_samples=1)
    return int(sorted_ids[sampled].item())


def load_tokens(path: Path, dtype_name: str) -> np.ndarray:
    dtype = np.uint16 if dtype_name == "uint16" else np.uint32
    data = np.memmap(path, dtype=dtype, mode="r")
    if data.ndim != 1 or len(data) == 0:
        raise ValueError(f"{path} does not contain any token ids")
    return data


def eval_loss(
    model: TransformerLM,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    eval_batches: int,
) -> float:
    losses = []
    with torch.no_grad():
        for _ in range(eval_batches):
            x, y = get_batch(dataset, batch_size, context_length, device)
            logits = model(x)
            loss = cross_entropy(logits.reshape(-1, logits.shape[-1]), y.reshape(-1))
            losses.append(float(loss.item()))
    return float(sum(losses) / len(losses))


def generate_sample(model: TransformerLM, tokenizer: Tokenizer, args: argparse.Namespace) -> str:
    ids = tokenizer.encode(args.prompt)
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            idx = torch.tensor([ids[-args.context_length :]], dtype=torch.long, device=args.device)
            logits = model(idx)[0, -1]
            probs = softmax(logits / args.temperature, dim=-1)
            ids.append(sample_top_p(probs, args.top_p))
    return tokenizer.decode(ids)


def main() -> int:
    args = parse_args()
    if args.num_threads is not None:
        torch.set_num_threads(args.num_threads)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    dataset = load_tokens(args.eval_data, args.dtype)
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
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    result: dict[str, object] = {
        "checkpoint": str(args.checkpoint),
        "eval_data": str(args.eval_data),
        "iteration": int(checkpoint.get("iteration", -1)),
        "eval_batches": args.eval_batches,
        "eval_loss": eval_loss(
            model,
            dataset,
            args.batch_size,
            args.context_length,
            args.device,
            args.eval_batches,
        ),
    }

    if args.vocab is not None and args.merges is not None:
        tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
        result["prompt"] = args.prompt
        result["sample"] = generate_sample(model, tokenizer, args)

    line = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    print(line)
    if args.log_file is not None:
        args.log_file.parent.mkdir(parents=True, exist_ok=True)
        args.log_file.write_text(line + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
