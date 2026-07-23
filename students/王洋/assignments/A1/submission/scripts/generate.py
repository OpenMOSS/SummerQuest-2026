#!/usr/bin/env python3
"""Generate text from a trained checkpoint with temperature and top-p sampling."""

from __future__ import annotations

import argparse
import contextlib
import json
import random
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from cs336_basics.model import TransformerLM
from cs336_basics.tokenizer import Tokenizer

from _common import atomic_write_json, load_json, load_tokenizer_artifact, resolve_device, utc_timestamp


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Autoregressively sample a Transformer LM checkpoint.")
    parser.add_argument("--tokenizer", type=Path, required=True, help="tokenizer directory or tokenizer.json")
    parser.add_argument("--config", type=Path, required=True, help="training/model JSON configuration")
    parser.add_argument("--checkpoint", type=Path, required=True, help="training checkpoint (.pt)")
    prompt = parser.add_mutually_exclusive_group(required=True)
    prompt.add_argument("--prompt", help="prompt text supplied directly on the command line")
    prompt.add_argument("--prompt-file", type=Path, help="UTF-8 file containing the prompt")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=1.0, help="zero selects greedy decoding")
    parser.add_argument("--top-p", type=float, default=0.9, help="nucleus probability mass in (0, 1]")
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, cuda:N, or mps")
    parser.add_argument("--stop-token", help="special token that ends generation (defaults to <|endoftext|>)")
    parser.add_argument("--ignore-stop-token", action="store_true", help="do not stop on a special token")
    parser.add_argument(
        "--allow-unbound-tokenizer",
        action="store_true",
        help="allow legacy configs/checkpoints with no tokenizer SHA-256 binding",
    )
    parser.add_argument("--output", type=Path, help="optional UTF-8 destination for generated text")
    parser.add_argument("--metadata-output", type=Path, help="optional JSON generation summary")
    return parser.parse_args()


def precision_context(device: torch.device, precision: str) -> contextlib.AbstractContextManager:
    if precision != "bfloat16":
        return contextlib.nullcontext()
    if device.type not in {"cpu", "cuda"}:
        raise ValueError(f"bfloat16 autocast is not supported by this script on {device.type}")
    return torch.autocast(device_type=device.type, dtype=torch.bfloat16)


def sample_next_token(logits: torch.Tensor, temperature: float, top_p: float) -> int:
    if temperature == 0:
        return int(torch.argmax(logits).item())

    probabilities = torch.softmax(logits.float() / temperature, dim=-1)
    if top_p < 1.0:
        sorted_probabilities, sorted_indices = torch.sort(probabilities, descending=True)
        cumulative = torch.cumsum(sorted_probabilities, dim=-1)
        remove = cumulative - sorted_probabilities >= top_p
        sorted_probabilities = sorted_probabilities.masked_fill(remove, 0)
        sorted_probabilities /= sorted_probabilities.sum()
        sampled_rank = torch.multinomial(sorted_probabilities, num_samples=1)
        return int(sorted_indices[sampled_rank].item())
    return int(torch.multinomial(probabilities, num_samples=1).item())


def find_stop_token_id(
    tokenizer: Tokenizer,
    special_tokens: list[str],
    requested: str | None,
) -> tuple[str | None, int | None]:
    if requested is not None:
        stop_token = requested
    elif "<|endoftext|>" in special_tokens:
        stop_token = "<|endoftext|>"
    else:
        stop_token = special_tokens[0] if special_tokens else None
    if stop_token is None:
        return None, None
    encoded = tokenizer.encode(stop_token)
    if len(encoded) != 1 or tokenizer.decode(encoded) != stop_token:
        raise ValueError(f"stop token {stop_token!r} is not a single tokenizer token")
    return stop_token, encoded[0]


def tokenizer_sha256_from_payload(payload: dict[str, Any] | None) -> dict[str, str] | None:
    """Extract a tokenizer checksum binding from a config or checkpoint payload."""

    if not isinstance(payload, dict):
        return None
    provenance = payload.get("data_provenance")
    candidates: list[Any] = [payload.get("tokenizer_sha256")]
    if isinstance(provenance, dict):
        candidates.append(provenance.get("tokenizer_sha256"))
        for split in ("train", "validation"):
            split_provenance = provenance.get(split)
            if isinstance(split_provenance, dict):
                candidates.append(split_provenance.get("tokenizer_sha256"))

    normalized_candidates = [
        {str(key): str(value) for key, value in candidate.items()}
        for candidate in candidates
        if isinstance(candidate, dict) and candidate
    ]
    if not normalized_candidates:
        return None
    expected = normalized_candidates[0]
    if any(candidate != expected for candidate in normalized_candidates[1:]):
        raise ValueError("training/validation provenance disagrees about the tokenizer SHA-256")
    return expected


def validate_tokenizer_binding(
    tokenizer_metadata: dict[str, Any],
    config: dict[str, Any],
    checkpoint: dict[str, Any] | Any,
    *,
    allow_unbound: bool,
) -> None:
    """Reject a same-sized tokenizer whose token-ID mapping differs from training."""

    actual = tokenizer_metadata.get("sha256")
    if not isinstance(actual, dict):
        raise ValueError("tokenizer metadata must contain vocab/merges SHA-256 checksums")
    actual = {str(key): str(value) for key, value in actual.items()}

    checkpoint_dict = checkpoint if isinstance(checkpoint, dict) else None
    candidates = [
        tokenizer_sha256_from_payload(config),
        tokenizer_sha256_from_payload(checkpoint_dict),
        tokenizer_sha256_from_payload(checkpoint_dict.get("config") if checkpoint_dict is not None else None),
    ]
    expected = next((candidate for candidate in candidates if candidate is not None), None)
    if expected is None:
        if not allow_unbound:
            raise ValueError(
                "config/checkpoint has no tokenizer SHA-256 binding; "
                "pass --allow-unbound-tokenizer only for an audited legacy run"
            )
        print(
            "warning: allowing an unbound legacy tokenizer; only vocab size can be verified",
            file=sys.stderr,
        )
        return
    if any(candidate is not None and candidate != expected for candidate in candidates):
        raise ValueError("config and checkpoint disagree about the training tokenizer SHA-256")
    if actual != expected:
        raise ValueError("tokenizer SHA-256 does not match the tokenizer bound to this training run")


def main() -> None:
    args = parse_args()
    if args.max_new_tokens < 0:
        raise ValueError("max-new-tokens must be non-negative")
    if args.temperature < 0:
        raise ValueError("temperature must be non-negative")
    if not 0 < args.top_p <= 1:
        raise ValueError("top-p must be in (0, 1]")

    prompt = args.prompt if args.prompt is not None else args.prompt_file.read_text(encoding="utf-8")
    tokenizer, tokenizer_metadata, _ = load_tokenizer_artifact(args.tokenizer)
    config = load_json(args.config)
    if "model" not in config or not isinstance(config["model"], dict):
        raise ValueError("configuration must contain a model object")
    if int(config["model"]["vocab_size"]) != len(tokenizer.vocab):
        raise ValueError("model vocab_size must exactly match the tokenizer artifact")

    device = resolve_device(args.device)
    random.seed(args.seed)
    np.random.seed(args.seed % (2**32))
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = TransformerLM(**config["model"], device=device)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    validate_tokenizer_binding(
        tokenizer_metadata,
        config,
        checkpoint,
        allow_unbound=args.allow_unbound_tokenizer,
    )
    checkpoint_model_config = checkpoint.get("config", {}).get("model") if isinstance(checkpoint, dict) else None
    if checkpoint_model_config is not None and checkpoint_model_config != config["model"]:
        raise ValueError("checkpoint model configuration does not match --config")
    state_dict = checkpoint["model"] if isinstance(checkpoint, dict) and "model" in checkpoint else checkpoint
    model.load_state_dict(state_dict)
    model.eval()

    special_tokens = tokenizer_metadata.get("special_tokens", [])
    stop_token, stop_token_id = find_stop_token_id(tokenizer, special_tokens, args.stop_token)
    seed_token_id = stop_token_id
    if args.ignore_stop_token:
        stop_token_id = None

    token_ids = tokenizer.encode(prompt)
    if not token_ids:
        if seed_token_id is None:
            raise ValueError("an empty prompt requires a tokenizer special token to seed generation")
        token_ids = [seed_token_id]
    prompt_tokens = len(token_ids)
    context_length = int(config["model"]["context_length"])
    precision = str(config.get("precision", "float32"))
    stopped_on_special = False
    start = time.perf_counter()

    with torch.inference_mode():
        for _ in range(args.max_new_tokens):
            context_ids = token_ids[-context_length:]
            inputs = torch.tensor(context_ids, dtype=torch.long, device=device).unsqueeze(0)
            with precision_context(device, precision):
                next_logits = model(inputs)[0, -1]
            next_token_id = sample_next_token(next_logits, args.temperature, args.top_p)
            token_ids.append(next_token_id)
            if stop_token_id is not None and next_token_id == stop_token_id:
                stopped_on_special = True
                break

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start
    generated_text = tokenizer.decode(token_ids)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(generated_text, encoding="utf-8")
    print(generated_text)

    generated_tokens = len(token_ids) - prompt_tokens
    summary = {
        "format": "cs336-generation-summary-v1",
        "created_at_utc": utc_timestamp(),
        "config_file": args.config.name,
        "checkpoint_file": args.checkpoint.name,
        "checkpoint_step": checkpoint.get("iteration") if isinstance(checkpoint, dict) else None,
        "checkpoint_processed_tokens": checkpoint.get("processed_tokens") if isinstance(checkpoint, dict) else None,
        "tokenizer_sha256": tokenizer_metadata.get("sha256"),
        "prompt": prompt,
        "prompt_tokens": prompt_tokens,
        "generated_tokens": generated_tokens,
        "total_tokens": len(token_ids),
        "max_new_tokens": args.max_new_tokens,
        "temperature": args.temperature,
        "top_p": args.top_p,
        "seed": args.seed,
        "stop_token": stop_token,
        "stopped_on_special_token": stopped_on_special,
        "stop_reason": "special_token" if stopped_on_special else "max_new_tokens",
        "elapsed_seconds": elapsed,
        "tokens_per_second": generated_tokens / elapsed if elapsed else None,
    }
    if args.metadata_output is not None:
        atomic_write_json(args.metadata_output, summary)
    print(json.dumps(summary, ensure_ascii=False, sort_keys=True), file=sys.stderr)


if __name__ == "__main__":
    main()
