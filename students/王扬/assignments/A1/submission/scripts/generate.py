from __future__ import annotations

import argparse
import os
from pathlib import Path

import torch

from cs336_basics.transformer import TransformerLM

from scripts.tokenizer_io import load_tokenizer
from scripts.utils import get_device, load_json, set_seed


def _sample_next_token(
    logits: torch.Tensor,
    *,
    temperature: float,
    top_p: float,
) -> int:
    if temperature <= 0:
        return int(torch.argmax(logits, dim=-1).item())

    probs = torch.softmax(logits / temperature, dim=-1)

    if top_p < 1.0:
        sorted_probs, sorted_idx = torch.sort(probs, descending=True)
        cum = torch.cumsum(sorted_probs, dim=-1)
        keep = cum <= top_p
        keep[..., 0] = True
        filtered = torch.where(keep, sorted_probs, torch.zeros_like(sorted_probs))
        filtered = filtered / filtered.sum(dim=-1, keepdim=True)
        next_sorted = torch.multinomial(filtered, num_samples=1).item()
        return int(sorted_idx[next_sorted].item())

    return int(torch.multinomial(probs, num_samples=1).item())


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_dir", required=True)
    parser.add_argument("--model_config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max_new_tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top_p", type=float, default=0.95)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    set_seed(args.seed)
    device = get_device(args.device)

    model_cfg = load_json(args.model_config)
    model = TransformerLM(
        vocab_size=int(model_cfg["vocab_size"]),
        context_length=int(model_cfg["context_length"]),
        d_model=int(model_cfg["d_model"]),
        num_layers=int(model_cfg["num_layers"]),
        num_heads=int(model_cfg["num_heads"]),
        d_ff=int(model_cfg["d_ff"]),
        rope_theta=float(model_cfg.get("rope_theta", 10000.0)),
        use_rope=bool(model_cfg.get("use_rope", True)),
        use_rmsnorm=bool(model_cfg.get("use_rmsnorm", True)),
        prenorm=bool(model_cfg.get("prenorm", True)),
        ffn_type=str(model_cfg.get("ffn_type", "swiglu")),
        eps=float(model_cfg.get("eps", 1e-5)),
    ).to(device)

    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model.load_state_dict(ckpt["model"])
    model.eval()

    special_tokens_path = Path(args.tokenizer_dir) / "special_tokens.json"
    special_tokens = []
    if special_tokens_path.exists():
        special_tokens = load_json(special_tokens_path)["special_tokens"]
    tokenizer = load_tokenizer(args.tokenizer_dir, special_tokens=special_tokens)

    ids = tokenizer.encode(args.prompt)
    if "<|endoftext|>" in tokenizer.special_tokens:
        stop_id = tokenizer.encode("<|endoftext|>")[0]
    else:
        stop_id = None

    x = torch.tensor(ids, dtype=torch.long, device=device).unsqueeze(0)

    for _ in range(args.max_new_tokens):
        x_in = x[:, -int(model_cfg["context_length"]) :]
        logits = model(x_in)
        next_id = _sample_next_token(
            logits[0, -1],
            temperature=args.temperature,
            top_p=args.top_p,
        )
        x = torch.cat([x, torch.tensor([[next_id]], dtype=torch.long, device=device)], dim=1)
        if stop_id is not None and next_id == stop_id:
            break

    out = tokenizer.decode(x[0].tolist())
    print(out)


if __name__ == "__main__":
    main()
