from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from cs336_basics.config import load_json_config
from cs336_basics.generation import generate
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.transformer import TransformerLM


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained checkpoint.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_json_config(args.config)
    device_spec = config.get("device", "auto")
    if device_spec == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_spec)
    model_config = config["model"]
    ablation = config.get("ablation", {})
    model = TransformerLM(
        **model_config,
        remove_rmsnorm=ablation.get("remove_rmsnorm", False),
        use_post_norm=ablation.get("use_post_norm", False),
        remove_rope=ablation.get("remove_rope", False),
        ffn_type=ablation.get("ffn_type"),
        device=device,
    )
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    tokenizer = Tokenizer.load(args.tokenizer)
    end_token_id = tokenizer.special_token_to_id.get("<|endoftext|>")
    text, token_ids, stop_reason = generate(
        model,
        tokenizer,
        args.prompt,
        args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        end_token_id=end_token_id,
        device=device,
    )
    result = {
        "generated_tokens": len(token_ids) - len(tokenizer.encode(args.prompt)),
        "stop_reason": stop_reason,
        "text": text,
    }
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(text)
    print(json.dumps({k: v for k, v in result.items() if k != "text"}, ensure_ascii=False))


if __name__ == "__main__":
    main()
