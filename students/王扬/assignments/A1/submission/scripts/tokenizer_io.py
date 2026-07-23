from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

from cs336_basics.tokenizer import BPETokenizer


@lru_cache(maxsize=1)
def _byte_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs, strict=True)}


@lru_cache(maxsize=1)
def _unicode_to_byte() -> dict[str, int]:
    m = _byte_to_unicode()
    return {v: k for k, v in m.items()}


def bytes_to_token_str(b: bytes) -> str:
    m = _byte_to_unicode()
    return "".join(m[x] for x in b)


def token_str_to_bytes(s: str) -> bytes:
    m = _unicode_to_byte()
    return bytes(m[ch] for ch in s)


def save_tokenizer(
    output_dir: str | os.PathLike,
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    vocab_out: dict[str, int] = {}
    for token_id, token_bytes in vocab.items():
        token_str: str | None
        try:
            token_str = token_bytes.decode("utf-8")
        except UnicodeDecodeError:
            token_str = None
        if token_str is None or token_str not in special_tokens:
            token_str = bytes_to_token_str(token_bytes)
        vocab_out[token_str] = token_id

    (out / "vocab.json").write_text(json.dumps(vocab_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    merges_lines = [f"{bytes_to_token_str(a)} {bytes_to_token_str(b)}" for a, b in merges]
    (out / "merges.txt").write_text("\n".join(merges_lines) + "\n", encoding="utf-8")

    (out / "special_tokens.json").write_text(
        json.dumps({"special_tokens": special_tokens}, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )


def load_tokenizer(
    tokenizer_dir: str | os.PathLike,
    special_tokens: list[str] | None = None,
) -> BPETokenizer:
    p = Path(tokenizer_dir)
    vocab_in = json.loads((p / "vocab.json").read_text(encoding="utf-8"))

    special = special_tokens
    if special is None and (p / "special_tokens.json").exists():
        special = json.loads((p / "special_tokens.json").read_text(encoding="utf-8"))["special_tokens"]
    if special is None:
        special = []

    vocab: dict[int, bytes] = {}
    for token_str, token_id in vocab_in.items():
        if token_str in special:
            vocab[token_id] = token_str.encode("utf-8")
        else:
            vocab[token_id] = token_str_to_bytes(token_str)

    merges: list[tuple[bytes, bytes]] = []
    for line in (p / "merges.txt").read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        a, b = line.split()
        merges.append((token_str_to_bytes(a), token_str_to_bytes(b)))

    return BPETokenizer(vocab=vocab, merges=merges, special_tokens=special)
