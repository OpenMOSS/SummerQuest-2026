from __future__ import annotations

import argparse
from pathlib import Path

import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import load_tokenizer


def choose_device(requested: str) -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def sample_top_p(logits: torch.Tensor, temperature: float, top_p: float) -> torch.Tensor:
    if temperature <= 0:
        return torch.argmax(logits, dim=-1, keepdim=True)

    logits = logits / temperature
    probs = torch.softmax(logits, dim=-1)

    if top_p >= 1.0:
        return torch.multinomial(probs, num_samples=1)

    sorted_probs, sorted_indices = torch.sort(probs, descending=True)
    cumulative = torch.cumsum(sorted_probs, dim=-1)
    mask = cumulative > top_p
    mask[..., 1:] = mask[..., :-1].clone()
    mask[..., 0] = False
    sorted_probs = sorted_probs.masked_fill(mask, 0.0)
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    sampled_sorted = torch.multinomial(sorted_probs, num_samples=1)
    return sorted_indices.gather(-1, sampled_sorted)


@torch.no_grad()
def generate(
    model: TransformerLM,
    prompt_ids: list[int],
    context_length: int,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    endoftext_id: int | None,
    device: str,
) -> list[int]:
    model.eval()
    generated = list(prompt_ids)

    for _ in range(max_new_tokens):
        context = generated[-context_length:]
        x = torch.tensor([context], dtype=torch.long, device=device)
        logits = model(x)[:, -1, :]
        next_id = int(sample_top_p(logits, temperature=temperature, top_p=top_p).item())
        generated.append(next_id)
        if endoftext_id is not None and next_id == endoftext_id:
            break

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate text from a trained Transformer LM checkpoint.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--prompt", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, required=True)
    parser.add_argument("--context-length", type=int, required=True)
    parser.add_argument("--d-model", type=int, required=True)
    parser.add_argument("--d-ff", type=int, required=True)
    parser.add_argument("--num-layers", type=int, required=True)
    parser.add_argument("--num-heads", type=int, required=True)
    parser.add_argument("--rope-theta", type=float, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    if not (0 < args.top_p <= 1):
        raise ValueError("--top-p must be in (0, 1].")

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = choose_device(args.device)
    tokenizer = load_tokenizer(args.tokenizer)
    prompt_ids = tokenizer.encode(args.prompt)
    if not prompt_ids:
        raise ValueError("Prompt encoded to zero tokens.")

    model = TransformerLM(
        vocab_size=args.vocab_size,
        context_length=args.context_length,
        d_model=args.d_model,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        d_ff=args.d_ff,
        rope_theta=args.rope_theta,
    ).to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"])

    endoftext_id = tokenizer.special_token_ids.get("<|endoftext|>")
    generated_ids = generate(
        model=model,
        prompt_ids=prompt_ids,
        context_length=args.context_length,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        endoftext_id=endoftext_id,
        device=device,
    )
    text = tokenizer.decode(generated_ids)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text, encoding="utf-8")
    print(f"Wrote {len(generated_ids) - len(prompt_ids)} generated tokens to {args.output}")


if __name__ == "__main__":
    main()
