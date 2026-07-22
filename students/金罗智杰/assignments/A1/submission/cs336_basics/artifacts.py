from __future__ import annotations

import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def save_tokenizer(
    path: str | Path,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "vocab": {str(token_id): token.hex() for token_id, token in vocab.items()},
        "merges": [[left.hex(), right.hex()] for left, right in merges],
        "special_tokens": special_tokens,
    }
    destination.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def load_tokenizer(path: str | Path) -> Tokenizer:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    vocab = {int(token_id): bytes.fromhex(token) for token_id, token in payload["vocab"].items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in payload["merges"]]
    return Tokenizer(vocab, merges, payload.get("special_tokens", []))
