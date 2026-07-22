from __future__ import annotations

import argparse
import json
import pickle
import sys
from contextlib import nullcontext
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from cs336_basics.nn import TransformerLM, softmax
from cs336_basics.tokenizer import Tokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate text from a trained TransformerLM checkpoint.")
    parser.add_argument("--checkpoint", required=True, help="Path to checkpoint saved by scripts/train_lm.py.")
    parser.add_argument("--config", required=True, help="Path to config.json saved by scripts/train_lm.py.")
    parser.add_argument(
        "--tokenizer",
        required=True,
        help="Pickle file containing (vocab, merges) or {'vocab': vocab, 'merges': merges}.",
    )
    parser.add_argument("--prompt", default="", help="Prompt text.")
    parser.add_argument("--prompt-file", default=None, help="Optional UTF-8 file containing the prompt.")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--special-token", action="append", default=None)
    parser.add_argument("--stop-token", default="<|endoftext|>")
    return parser.parse_args()


def load_tokenizer(path: str | Path, special_tokens: list[str] | None) -> Tokenizer:
    with open(path, "rb") as f:
        payload = pickle.load(f)
    if isinstance(payload, dict):
        vocab = payload["vocab"]
        merges = payload["merges"]
        special_tokens = payload.get("special_tokens", special_tokens)
    else:
        vocab, merges = payload
    if special_tokens is None:
        special_tokens = ["<|endoftext|>"]
    vocab_bytes = set(vocab.values())
    special_tokens = [token for token in special_tokens if token.encode("utf-8") in vocab_bytes]
    return Tokenizer(vocab, merges, special_tokens)


def build_model(config: dict, device: str) -> TransformerLM:
    model = TransformerLM(
        vocab_size=int(config["vocab_size"]),
        context_length=int(config["context_length"]),
        d_model=int(config["d_model"]),
        num_layers=int(config["num_layers"]),
        num_heads=int(config["num_heads"]),
        d_ff=int(config["d_ff"]),
        rope_theta=float(config["rope_theta"]),
        norm_type=config.get("norm_type", "rmsnorm"),
        norm_position=config.get("norm_position", "pre"),
        pos_emb=config.get("pos_emb", "rope"),
        ffn_type=config.get("ffn_type", "swiglu"),
    )
    return model.to(device)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)
    probs = softmax(logits / temperature, dim=-1)
    if top_p < 1.0:
        if not 0.0 < top_p <= 1.0:
            raise ValueError("top_p must be in (0, 1]")
        sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
        cumulative = torch.cumsum(sorted_probs, dim=-1)
        remove = cumulative > top_p
        remove[..., 1:] = remove[..., :-1].clone()
        remove[..., 0] = False
        sorted_probs = sorted_probs.masked_fill(remove, 0.0)
        sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
        sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
        return sorted_indices.gather(-1, sampled_sorted)
    return torch.multinomial(probs, num_samples=1)


@torch.no_grad()
def generate(
    model: TransformerLM,
    tokenizer: Tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    stop_token: str | None,
    device: str,
    use_amp: bool,
) -> list[int]:
    model.eval()
    token_ids = tokenizer.encode(prompt)
    generated = torch.tensor([token_ids], dtype=torch.long, device=device)
    stop_id = None
    if stop_token is not None:
        stop_bytes = stop_token.encode("utf-8")
        stop_id = tokenizer.byte_to_id.get(stop_bytes)

    for _ in range(max_new_tokens):
        context = generated[:, -model.context_length :]
        amp_context = torch.autocast(device_type="cuda", dtype=torch.bfloat16) if use_amp else nullcontext()
        with amp_context:
            logits = model(context)[:, -1, :]
        next_token = sample_next_token(logits, temperature, top_p)
        generated = torch.cat([generated, next_token], dim=-1)
        if stop_id is not None and int(next_token.item()) == stop_id:
            break
    return generated[0].tolist()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    prompt = args.prompt
    if args.prompt_file is not None:
        prompt = Path(args.prompt_file).read_text(encoding="utf-8")

    config = json.loads(Path(args.config).read_text(encoding="utf-8"))
    tokenizer = load_tokenizer(args.tokenizer, args.special_token)
    model = build_model(config, args.device)
    checkpoint = torch.load(args.checkpoint, map_location=args.device)
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)

    use_amp = args.device.startswith("cuda") and args.dtype == "bfloat16"
    ids = generate(
        model,
        tokenizer,
        prompt,
        args.max_new_tokens,
        args.temperature,
        args.top_p,
        args.stop_token,
        args.device,
        use_amp,
    )
    print(tokenizer.decode(ids), end="")


if __name__ == "__main__":
    main()
