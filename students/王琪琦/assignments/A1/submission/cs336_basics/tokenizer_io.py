from __future__ import annotations

import json
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


def save_tokenizer(tokenizer: Tokenizer, output_dir: str | Path) -> None:
    directory = Path(output_dir)
    directory.mkdir(parents=True, exist_ok=True)
    vocabulary = {str(token_id): token.hex() for token_id, token in tokenizer.vocab.items()}
    merges = [[left.hex(), right.hex()] for left, right in tokenizer.merges]
    (directory / "vocab.json").write_text(json.dumps(vocabulary, indent=2), encoding="utf-8")
    (directory / "merges.json").write_text(json.dumps(merges, indent=2), encoding="utf-8")
    metadata = {"special_tokens": tokenizer.special_tokens}
    (directory / "tokenizer_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def load_tokenizer(input_dir: str | Path) -> Tokenizer:
    directory = Path(input_dir)
    vocabulary_json = json.loads((directory / "vocab.json").read_text(encoding="utf-8"))
    merges_json = json.loads((directory / "merges.json").read_text(encoding="utf-8"))
    metadata = json.loads((directory / "tokenizer_config.json").read_text(encoding="utf-8"))
    vocabulary = {int(token_id): bytes.fromhex(token) for token_id, token in vocabulary_json.items()}
    merges = [(bytes.fromhex(left), bytes.fromhex(right)) for left, right in merges_json]
    return Tokenizer(vocabulary, merges, metadata.get("special_tokens"))
