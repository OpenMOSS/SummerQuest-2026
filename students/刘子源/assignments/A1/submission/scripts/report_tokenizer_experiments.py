# 看 tokenizer 压缩率的小脚本，拿前几篇文档算个够用的数。

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.tokenizer import Tokenizer  # noqa: E402


def load_documents(path: Path, delimiter: str, count: int) -> list[str]:
    docs: list[str] = []
    pending = ""
    with path.open(encoding="utf-8") as input_file:
        while len(docs) < count:
            chunk = input_file.read(1024 * 1024)
            if not chunk:
                break
            pending += chunk
            pieces = pending.split(delimiter)
            pending = pieces.pop()
            docs.extend(piece for piece in pieces if piece)
    if len(docs) < count and pending:
        docs.append(pending)
    if len(docs) < count:
        raise ValueError(f"{path} contains only {len(docs)} non-empty documents")
    return docs[:count]


def compression(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float | int]:
    byte_num = sum(len(doc.encode("utf-8")) for doc in documents)
    token_num = sum(len(tokenizer.encode(doc)) for doc in documents)
    return {
        "documents": len(documents),
        "bytes": byte_num,
        "tokens": token_num,
        "bytes_per_token": byte_num / token_num,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tinystories", type=Path, required=True)
    parser.add_argument("--owt", type=Path, required=True)
    parser.add_argument("--tinystories-vocab", type=Path, required=True)
    parser.add_argument("--tinystories-merges", type=Path, required=True)
    parser.add_argument("--owt-vocab", type=Path, required=True)
    parser.add_argument("--owt-merges", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--delimiter", default="<|endoftext|>")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.documents < 1:
        parser.error("--documents must be positive")

    special_tokens = [args.delimiter]
    tiny_tokenizer = Tokenizer.from_files(
        args.tinystories_vocab, args.tinystories_merges, special_tokens
    )
    owt_tokenizer = Tokenizer.from_files(args.owt_vocab, args.owt_merges, special_tokens)
    tiny_documents = load_documents(args.tinystories, args.delimiter, args.documents)
    owt_documents = load_documents(args.owt, args.delimiter, args.documents)
    report = {
        "tinystories_with_tinystories_tokenizer": compression(tiny_tokenizer, tiny_documents),
        "owt_with_owt_tokenizer": compression(owt_tokenizer, owt_documents),
        "owt_with_tinystories_tokenizer": compression(tiny_tokenizer, owt_documents),
    }
    json_text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json_text, encoding="utf-8")
    print(json_text, end="")


if __name__ == "__main__":
    main()
