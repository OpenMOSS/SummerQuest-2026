#!/usr/bin/env python3
"""Generate text from a trained checkpoint using temperature and top-p sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

from cs336_basics.config import apply_overrides, load_json_config, project_root, resolve_project_path
from cs336_basics.experiment import (
    build_transformer,
    canonical_model_config,
    checkpoint_model_config,
    extract_model_state,
    load_checkpoint_payload,
)
from cs336_basics.generation import generate_text
from cs336_basics.tokenizer import Tokenizer
from cs336_basics.training import resolve_device, resolve_dtype, set_seed


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--tokenizer-dir", required=True, type=Path)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--allow-tokenizer-mismatch", action="store_true")
    parser.add_argument("--set", dest="overrides", action="append", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = project_root()
    config = apply_overrides(load_json_config(args.config), args.overrides)
    set_seed(args.seed)
    device = resolve_device(str(config.get("device", "auto")))
    compute_dtype = resolve_dtype(str(config.get("dtype", "float32")), device)
    parameter_dtype = resolve_dtype(str(config.get("parameter_dtype", "float32")), device)
    amp_enabled = bool(config.get("amp", device.type in {"cuda", "mps"})) and compute_dtype != torch.float32

    tokenizer_dir = resolve_project_path(args.tokenizer_dir, root=root)
    checkpoint_path = resolve_project_path(args.checkpoint, root=root)
    assert tokenizer_dir is not None and checkpoint_path is not None
    tokenizer = Tokenizer.from_directory(tokenizer_dir)
    payload = load_checkpoint_payload(checkpoint_path, map_location="cpu")
    model_config = checkpoint_model_config(payload, config["model"])
    if canonical_model_config(model_config) != canonical_model_config(config["model"]):
        print(
            "warning: using architecture stored in the checkpoint instead of the external config",
            file=sys.stderr,
        )
    model = build_transformer(model_config, device=device, dtype=parameter_dtype)
    model.load_state_dict(extract_model_state(payload))

    if len(tokenizer.vocab) != int(model_config["vocab_size"]):
        raise ValueError(
            f"tokenizer vocabulary has {len(tokenizer.vocab)} entries but checkpoint model expects "
            f"{model_config['vocab_size']}"
        )
    checkpoint_data = payload.get("config", {}).get("data", {})
    expected_tokenizer_dir = checkpoint_data.get("tokenizer_dir") if isinstance(checkpoint_data, dict) else None
    expected_tokenizer_path = (
        resolve_project_path(expected_tokenizer_dir, root=root) if expected_tokenizer_dir is not None else None
    )
    if expected_tokenizer_path is not None and tokenizer_dir != expected_tokenizer_path:
        if not args.allow_tokenizer_mismatch:
            raise ValueError(
                "tokenizer directory differs from the checkpoint config; "
                "pass --allow-tokenizer-mismatch to override intentionally"
            )
        print("warning: using an externally supplied tokenizer", file=sys.stderr)

    eos_bytes = b"<|endoftext|>"
    eos_token_id = next((token_id for token_id, value in tokenizer.vocab.items() if value == eos_bytes), None)
    text, token_ids = generate_text(
        model,
        tokenizer,
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        context_length=int(model_config["context_length"]),
        device=device,
        temperature=args.temperature,
        top_p=args.top_p,
        eos_token_id=eos_token_id,
        seed=args.seed,
        amp_dtype=compute_dtype,
        amp_enabled=amp_enabled,
    )
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_iteration": int(payload.get("iteration", -1)),
        "prompt": args.prompt,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "num_prompt_tokens": len(tokenizer.encode(args.prompt)),
        "num_total_tokens": len(token_ids),
        "text": text,
    }
    rendered = json.dumps(result, ensure_ascii=False, indent=2, sort_keys=True)
    if args.output:
        output_path = resolve_project_path(args.output, root=root)
        assert output_path is not None
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered + "\n", encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
