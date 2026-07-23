import argparse
import json
from pathlib import Path
import random
import time

from cs336_basics.tokenizer import Tokenizer


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure tokenizer compression and throughput")
    parser.add_argument("input", type=Path)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--documents", type=int, default=10)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, ["<|endoftext|>"])
    text = args.input.read_text(encoding="utf-8")
    documents = [document for document in text.split("<|endoftext|>") if document]
    sample = random.Random(args.seed).sample(documents, min(args.documents, len(documents)))
    sample_text = "<|endoftext|>".join(sample)
    input_bytes = len(sample_text.encode("utf-8"))
    start = time.perf_counter()
    ids = tokenizer.encode(sample_text)
    elapsed = time.perf_counter() - start
    result = {
        "source": args.input.name,
        "sample_documents": len(sample),
        "input_bytes": input_bytes,
        "tokens": len(ids),
        "compression_bytes_per_token": input_bytes / len(ids),
        "elapsed_sec": elapsed,
        "throughput_bytes_per_sec": input_bytes / elapsed,
    }
    encoded = json.dumps(result, indent=2)
    print(encoded)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(encoded, encoding="utf-8")


if __name__ == "__main__":
    main()
