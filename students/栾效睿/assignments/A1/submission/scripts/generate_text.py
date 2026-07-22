from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch

if __package__ in {None, ""}:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.bpe_tokenizer import BPETokenizer
from cs336_basics.softmax import softmax
from cs336_basics.transformer_lm import TransformerLM
from scripts.experiment_utils import append_jsonl, load_json, parse_dtype, project_path, seed_all, select_device, sha256
from scripts.train import resolve_config


def build_model(cfg: dict[str, Any], device: torch.device) -> TransformerLM:
    m = cfg["model"]
    return TransformerLM(
        d_model=m["d_model"],
        num_heads=m["num_heads"],
        vocab_size=m["vocab_size"],
        num_layers=m["num_layers"],
        max_seq_len=m["context_length"],
        d_ff=m["d_ff"],
        theta=m["theta"],
        use_rope=m["use_rope"],
        eps=m["eps"],
        device=device,
        dtype=parse_dtype(cfg["training"].get("dtype")),
        norm_mode=m["norm_mode"],
        ffn_type=m["ffn_type"],
    )


def sample_token(logits: torch.Tensor, temperature: float, top_p: float, generator: torch.Generator) -> int:
    if temperature <= 0 or not 0 < top_p <= 1:
        raise ValueError("temperature must be > 0 and top_p must be in (0, 1]")
    probs = softmax((logits.float() / temperature).cpu(), dim=-1)
    sorted_probs, sorted_ids = torch.sort(probs, descending=True)
    if top_p < 1:
        remove = torch.cumsum(sorted_probs, dim=-1) > top_p
        remove[1:] = remove[:-1].clone()
        remove[0] = False
        sorted_probs[remove] = 0
        sorted_probs /= sorted_probs.sum()
    return int(sorted_ids[torch.multinomial(sorted_probs, 1, generator=generator)].item())


@torch.inference_mode()
def generate(model: TransformerLM, tokenizer: BPETokenizer, prompt: str, max_new_tokens: int, temperature: float, top_p: float, seed: int, eos_token: str, device: torch.device) -> tuple[str, int, str]:
    ids = tokenizer.encode(prompt)
    if not ids:
        raise ValueError("prompt produced no tokens")
    eos_id = tokenizer.reverse_vocab.get(eos_token.encode("utf-8"))
    generator = torch.Generator(device="cpu").manual_seed(seed)
    new_ids: list[int] = []
    stop_reason = "max_new_tokens"
    for _ in range(max_new_tokens):
        x = torch.tensor([ids[-model.context_length :]], dtype=torch.long, device=device)
        next_id = sample_token(model(x)[0, -1], temperature, top_p, generator)
        if next_id == eos_id:
            stop_reason = "eos_token"
            break
        ids.append(next_id)
        new_ids.append(next_id)
    return tokenizer.decode(new_ids), len(new_ids), stop_reason


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate text from a trained CS336 checkpoint.")
    parser.add_argument("--config", required=True, type=Path, help="Generation config in configs/.")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--prompt")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    cfg = load_json(args.config)
    train_cfg = resolve_config(load_json(cfg["training_config"]))
    checkpoint = project_path(args.checkpoint or cfg["checkpoint"]).resolve(strict=True)
    prompt = args.prompt if args.prompt is not None else str(cfg["prompt"])
    max_new_tokens = args.max_new_tokens if args.max_new_tokens is not None else int(cfg.get("max_new_tokens", 256))
    temperature = args.temperature if args.temperature is not None else float(cfg.get("temperature", 0.8))
    top_p = args.top_p if args.top_p is not None else float(cfg.get("top_p", 0.9))
    seed = args.seed if args.seed is not None else int(cfg.get("seed", 1337))
    output = project_path(args.output or cfg.get("output", "logs/generation_samples.jsonl"))
    device = select_device(args.device or cfg.get("device") or train_cfg["training"].get("device"))
    seed_all(seed)

    tokenizer = BPETokenizer.from_files(
        train_cfg["data"]["vocab_path"],
        train_cfg["data"]["merges_path"],
        special_tokens=train_cfg["data"]["special_tokens"],
    )
    eos_token = cfg.get("eos_token", "<|endoftext|>")
    eos_id = tokenizer.reverse_vocab.get(eos_token.encode("utf-8"))
    if eos_id is None or eos_id >= train_cfg["model"]["vocab_size"]:
        raise ValueError("configured eos_token is not present in the model vocabulary")
    if max(tokenizer.vocab) >= train_cfg["model"]["vocab_size"]:
        raise ValueError("tokenizer vocabulary is larger than the configured model vocabulary")
    model = build_model(train_cfg, device)
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state"] if isinstance(payload, dict) and "model_state" in payload else payload)
    model.eval()

    reply, generated_tokens, stop_reason = generate(
        model, tokenizer, prompt, max_new_tokens, temperature, top_p, seed, eos_token, device
    )
    record = {
        "timestamp_utc": datetime.now(UTC).isoformat(),
        "training_config": str(project_path(cfg["training_config"])),
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": sha256(checkpoint),
        "prompt": prompt,
        "reply": reply,
        "generated_tokens": generated_tokens,
        "stop_reason": stop_reason,
        "temperature": temperature,
        "top_p": top_p,
        "seed": seed,
        "device": str(device),
    }
    append_jsonl(output, record)
    print(reply)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
