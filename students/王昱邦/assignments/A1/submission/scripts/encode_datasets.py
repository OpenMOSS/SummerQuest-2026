from __future__ import annotations

import argparse
import json
import os
import time
from collections.abc import Iterator
from pathlib import Path

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
END_OF_TEXT = "<|endoftext|>"
END_OF_TEXT_BYTES = END_OF_TEXT.encode("utf-8")
OUTPUT_ROOT = PROJECT_ROOT / "artifacts" / "tokenized_datasets"

TINYSTORIES_TOKENIZER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "tinystories_bpe"
    / "train"
    / "vocab_10000"
    / "tinystories_train_10000_v1"
)
OWT_TOKENIZER_DIR = (
    PROJECT_ROOT
    / "artifacts"
    / "owt_bpe"
    / "owt-train"
    / "vocab_32000"
    / "owt_train_32000_v1"
)

DATASET_CONFIGS = {
    "tinystories-train": (
        PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-train.txt",
        TINYSTORIES_TOKENIZER_DIR,
        "tinystories_10k",
    ),
    "tinystories-valid": (
        PROJECT_ROOT / "data" / "TinyStoriesV2-GPT4-valid.txt",
        TINYSTORIES_TOKENIZER_DIR,
        "tinystories_10k",
    ),
    "owt-train": (
        PROJECT_ROOT / "data" / "owt_train.txt",
        OWT_TOKENIZER_DIR,
        "owt_32k",
    ),
    "owt-valid": (
        PROJECT_ROOT / "data" / "owt_valid.txt",
        OWT_TOKENIZER_DIR,
        "owt_32k",
    ),
}


def iter_document_bytes(
    input_path: Path,
    read_size: int = 4 * 1024 * 1024,
) -> Iterator[tuple[bytes, int, bool]]:
    """Yield (document, consumed input bytes, ended_by_special_token)."""
    buffer = b""

    with input_path.open("rb") as input_file:
        while chunk := input_file.read(read_size):
            buffer += chunk

            while True:
                delimiter_index = buffer.find(END_OF_TEXT_BYTES)
                if delimiter_index == -1:
                    break

                consumed_bytes = delimiter_index + len(END_OF_TEXT_BYTES)
                document = buffer[:delimiter_index]
                buffer = buffer[consumed_bytes:]
                yield document, consumed_bytes, True

        if buffer:
            yield buffer, len(buffer), False


def load_tokenizer(tokenizer_dir: Path) -> Tokenizer:
    return Tokenizer.from_files(
        str(tokenizer_dir / "vocab.pkl"),
        str(tokenizer_dir / "merges.pkl"),
        special_tokens=[END_OF_TEXT],
    )


def encode_dataset(
    dataset_name: str,
    input_path: Path,
    tokenizer: Tokenizer,
    tokenizer_name: str,
    output_root: Path,
) -> dict[str, int | float | str]:
    output_root.mkdir(parents=True, exist_ok=True)
    output_path = output_root / f"{dataset_name}.uint16.bin"
    metadata_path = output_root / f"{dataset_name}.metadata.json"
    temporary_output_path = output_path.with_suffix(output_path.suffix + ".partial")
    temporary_metadata_path = metadata_path.with_suffix(metadata_path.suffix + ".partial")

    for path in (output_path, metadata_path, temporary_output_path, temporary_metadata_path):
        if path.exists():
            raise FileExistsError(f"Refusing to overwrite existing output: {path}")

    input_bytes = input_path.stat().st_size
    special_token_id = tokenizer.special_token_to_id[END_OF_TEXT]
    token_count = 0
    document_count = 0
    special_token_count = 0
    maximum_token_id = -1
    start_time = time.perf_counter()

    try:
        with temporary_output_path.open("wb") as output_file, tqdm(
            total=input_bytes,
            desc=dataset_name,
            unit="B",
            unit_scale=True,
            unit_divisor=1024,
            dynamic_ncols=True,
        ) as progress:
            for document_bytes, consumed_bytes, ended_by_special_token in iter_document_bytes(
                input_path
            ):
                document = document_bytes.decode("utf-8")
                token_ids = tokenizer.encode(document)

                if token_ids:
                    token_array = np.asarray(token_ids, dtype="<u2")
                    token_array.tofile(output_file)
                    token_count += token_array.size
                    maximum_token_id = max(maximum_token_id, int(token_array.max()))

                if ended_by_special_token:
                    np.asarray([special_token_id], dtype="<u2").tofile(output_file)
                    token_count += 1
                    special_token_count += 1
                    maximum_token_id = max(maximum_token_id, special_token_id)

                document_count += 1
                progress.update(consumed_bytes)
                progress.set_postfix(tokens=f"{token_count:,}", refresh=False)

            output_file.flush()
            os.fsync(output_file.fileno())

        elapsed_seconds = time.perf_counter() - start_time
        if maximum_token_id >= 2**16:
            raise ValueError(
                f"Token ID {maximum_token_id} does not fit in uint16 for {dataset_name}."
            )
        expected_output_bytes = token_count * np.dtype("<u2").itemsize
        if temporary_output_path.stat().st_size != expected_output_bytes:
            raise RuntimeError("Encoded output size does not match the token count.")

        metadata: dict[str, int | float | str] = {
            "dataset": dataset_name,
            "input_path": str(input_path),
            "input_bytes": input_bytes,
            "output_path": str(output_path),
            "output_format": "raw NumPy-compatible binary",
            "dtype": "<u2",
            "tokenizer": tokenizer_name,
            "vocabulary_size": len(tokenizer.vocab),
            "token_count": token_count,
            "document_count": document_count,
            "special_token": END_OF_TEXT,
            "special_token_id": special_token_id,
            "special_token_count": special_token_count,
            "maximum_token_id": maximum_token_id,
            "bytes_per_token": input_bytes / token_count,
            "elapsed_seconds": elapsed_seconds,
            "bytes_per_second": input_bytes / elapsed_seconds,
        }
        temporary_metadata_path.write_text(
            json.dumps(metadata, indent=2), encoding="utf-8"
        )

        os.replace(temporary_output_path, output_path)
        os.replace(temporary_metadata_path, metadata_path)
        return metadata
    except BaseException:
        temporary_output_path.unlink(missing_ok=True)
        temporary_metadata_path.unlink(missing_ok=True)
        raise


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Stream datasets through trained tokenizers into uint16 files."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        choices=list(DATASET_CONFIGS),
        default=list(DATASET_CONFIGS),
    )
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    loaded_tokenizers: dict[Path, Tokenizer] = {}
    all_metadata = []

    for dataset_name in args.datasets:
        input_path, tokenizer_dir, tokenizer_name = DATASET_CONFIGS[dataset_name]
        if not input_path.exists():
            raise FileNotFoundError(input_path)
        if tokenizer_dir not in loaded_tokenizers:
            loaded_tokenizers[tokenizer_dir] = load_tokenizer(tokenizer_dir)

        metadata = encode_dataset(
            dataset_name=dataset_name,
            input_path=input_path,
            tokenizer=loaded_tokenizers[tokenizer_dir],
            tokenizer_name=tokenizer_name,
            output_root=args.output_dir,
        )
        all_metadata.append(metadata)
        print(json.dumps(metadata, indent=2))

    print(f"Completed {len(all_metadata)} dataset(s). Output: {args.output_dir}")


if __name__ == "__main__":
    main()
