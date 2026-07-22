import argparse
import json
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer


def sample(logits, temperature: float, top_p: float) -> int:
    logits = logits / temperature
    probabilities = torch.softmax(logits, dim=-1)
    sorted_probs, sorted_ids = torch.sort(probabilities, descending=True)
    keep = sorted_probs.cumsum(dim=-1) - sorted_probs <= top_p
    filtered = sorted_probs * keep
    filtered = filtered / filtered.sum()
    return sorted_ids[torch.multinomial(filtered, 1)].item()


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from an A1 checkpoint")
    parser.add_argument("config", type=Path)
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--prompt", default="Once upon a time")
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path, help="Optional JSON file with sampling settings and generated text")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()
    if args.temperature <= 0:
        raise ValueError("temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise ValueError("top_p must be in (0, 1]")
    torch.manual_seed(args.seed)
    config = json.loads(args.config.read_text())
    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    model = TransformerLM(
        config["vocab_size"], config["context_length"], config["d_model"], config["num_layers"],
        config["num_heads"], config["d_ff"], config.get("rope_theta", 10000.0),
        config.get("norm_style", "pre"), config.get("use_rope", True), config.get("ffn_variant", "swiglu"),
    ).to(args.device)
    state = torch.load(args.checkpoint, map_location=args.device)
    model.load_state_dict(state["model"])
    model.eval()
    ids = tokenizer.encode(args.prompt)
    prompt_tokens = len(ids)
    end_id = tokenizer.special_to_id.get("<|endoftext|>")
    stopped_on_eot = False
    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            context = torch.tensor([ids[-config["context_length"] :]], device=args.device)
            next_id = sample(model(context)[0, -1], args.temperature, args.top_p)
            ids.append(next_id)
            if next_id == end_id:
                stopped_on_eot = True
                break
    text = tokenizer.decode(ids)
    print(text)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        result = {
            "prompt": args.prompt,
            "seed": args.seed,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "generated_tokens": len(ids) - prompt_tokens,
            "stopped_on_eot": stopped_on_eot,
            "text": text,
        }
        args.output.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
