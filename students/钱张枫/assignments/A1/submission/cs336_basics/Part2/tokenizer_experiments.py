from __future__ import annotations

import argparse
import codecs
import json
import time
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from cs336_basics.Part2.tokenizer import Tokenizer


"""
Tokenizer 实验脚本。

重要约束：
1. 本脚本直接复用 cs336_basics.Part2.tokenizer.Tokenizer，不修改 tokenizer.py。
2. 这里没有复制 tokenizer 源码；所有编码/解码行为都来自已实现的 Tokenizer。
3. 如果后续为了实验性能需要改写 tokenizer 算法，应在本文件中新增独立实验实现，并在注释中明确标出差异。
"""


DEFAULT_SPECIAL_TOKENS = ["<|endoftext|>"]
DEFAULT_SAMPLE_DOCUMENTS = 10
DEFAULT_BENCHMARK_BYTES = 2 * 1024 * 1024
DEFAULT_CHUNK_BYTES = 1 * 1024 * 1024
DEFAULT_WRITE_BUFFER_TOKENS = 1_000_000
PILE_BYTES = 825 * 1000**3
UINT16_MAX = np.iinfo(np.uint16).max


@dataclass(frozen=True)
class TokenizerBundle:
    name: str
    tokenizer: Tokenizer
    vocab_path: Path
    merges_path: Path


@dataclass(frozen=True)
class DatasetPaths:
    train: Path
    dev: Path


def load_tokenizer(name: str, artifact_dir: Path) -> TokenizerBundle:
    vocab_path = artifact_dir / "vocab.json"
    merges_path = artifact_dir / "merges.json"
    tokenizer = Tokenizer.from_files(
        vocab_filepath=str(vocab_path),
        merges_filepath=str(merges_path),
        special_tokens=DEFAULT_SPECIAL_TOKENS,
    )
    return TokenizerBundle(name=name, tokenizer=tokenizer, vocab_path=vocab_path, merges_path=merges_path)


def iter_documents(input_path: Path, delimiter: bytes = b"<|endoftext|>", chunk_size: int = 8 * 1024 * 1024) -> Iterator[str]:
    """按 special token 文档边界流式产出文档文本，不把整个语料读入内存。"""
    remainder = b""
    with input_path.open("rb") as input_file:
        while chunk := input_file.read(chunk_size):
            pieces = (remainder + chunk).split(delimiter)
            remainder = pieces.pop()
            for piece in pieces:
                if piece:
                    yield piece.decode("utf-8", errors="replace")

    if remainder:
        yield remainder.decode("utf-8", errors="replace")


def sample_first_documents(input_path: Path, sample_size: int) -> list[str]:
    """取前 N 个非空文档作为可复现实验样本。"""
    documents: list[str] = []
    for document in iter_documents(input_path):
        documents.append(document)
        if len(documents) == sample_size:
            break
    if len(documents) < sample_size:
        raise ValueError(f"{input_path} only contains {len(documents)} non-empty documents")
    return documents


def compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> dict[str, float | int]:
    total_bytes = 0
    total_tokens = 0
    for document in documents:
        total_bytes += len(document.encode("utf-8"))
        total_tokens += len(tokenizer.encode(document))

    if total_tokens == 0:
        raise ValueError("Cannot compute compression ratio for zero tokens")

    return {
        "bytes": total_bytes,
        "tokens": total_tokens,
        "bytes_per_token": total_bytes / total_tokens,
    }


def read_text_prefix(input_path: Path, target_bytes: int) -> str:
    """读取约 target_bytes 大小的 UTF-8 文本前缀，用于吞吐 benchmark。"""
    with input_path.open("rb") as input_file:
        data = input_file.read(target_bytes)
    return data.decode("utf-8", errors="ignore")


def benchmark_throughput(tokenizer: Tokenizer, text: str) -> dict[str, float | int]:
    input_bytes = len(text.encode("utf-8"))
    start = time.perf_counter()
    token_ids = tokenizer.encode(text)
    elapsed_seconds = time.perf_counter() - start
    if elapsed_seconds <= 0:
        raise ValueError("Benchmark elapsed time must be positive")

    return {
        "bytes": input_bytes,
        "tokens": len(token_ids),
        "seconds": elapsed_seconds,
        "bytes_per_second": input_bytes / elapsed_seconds,
        "tokens_per_second": len(token_ids) / elapsed_seconds,
    }


def hours_for_bytes(total_bytes: int, bytes_per_second: float) -> float:
    return total_bytes / bytes_per_second / 3600


def iter_utf8_text_chunks(
    input_path: Path,
    chunk_bytes: int,
    show_progress: bool,
    description: str,
) -> Iterator[str]:
    """按真实输入 bytes 显示进度，并用增量解码避免 chunk 边界切坏 UTF-8 字符。"""
    decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
    total_bytes = input_path.stat().st_size

    with tqdm(
        total=total_bytes,
        desc=description,
        unit="B",
        unit_scale=True,
        disable=not show_progress,
    ) as progress:
        with input_path.open("rb") as input_file:
            while raw_chunk := input_file.read(chunk_bytes):
                progress.update(len(raw_chunk))
                text_chunk = decoder.decode(raw_chunk, final=False)
                if text_chunk:
                    yield text_chunk

        tail = decoder.decode(b"", final=True)
        if tail:
            yield tail


def run_report(args: argparse.Namespace) -> dict[str, Any]:
    tiny_tokenizer = load_tokenizer("tinystories_10k", args.tinystories_artifacts)
    owt_tokenizer = load_tokenizer("owt_32k", args.owt_artifacts)

    tiny_documents = sample_first_documents(args.tinystories_train, args.sample_documents)
    owt_documents = sample_first_documents(args.owt_train, args.sample_documents)

    tiny_on_tiny = compression_ratio(tiny_tokenizer.tokenizer, tiny_documents)
    owt_on_owt = compression_ratio(owt_tokenizer.tokenizer, owt_documents)
    tiny_on_owt = compression_ratio(tiny_tokenizer.tokenizer, owt_documents)

    tiny_benchmark_text = read_text_prefix(args.tinystories_train, args.benchmark_bytes)
    owt_benchmark_text = read_text_prefix(args.owt_train, args.benchmark_bytes)
    tiny_throughput = benchmark_throughput(tiny_tokenizer.tokenizer, tiny_benchmark_text)
    owt_throughput = benchmark_throughput(owt_tokenizer.tokenizer, owt_benchmark_text)

    report = {
        "sample_documents": args.sample_documents,
        "benchmark_bytes": args.benchmark_bytes,
        "compression": {
            "tinystories_sample_with_tinystories_10k": tiny_on_tiny,
            "owt_sample_with_owt_32k": owt_on_owt,
            "owt_sample_with_tinystories_10k": tiny_on_owt,
        },
        "throughput": {
            "tinystories_10k_on_tinystories_prefix": tiny_throughput,
            "owt_32k_on_owt_prefix": owt_throughput,
        },
        "pile_estimate": {
            "pile_bytes": PILE_BYTES,
            "hours_using_owt_32k_throughput": hours_for_bytes(
                PILE_BYTES,
                float(owt_throughput["bytes_per_second"]),
            ),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report_path = args.output_dir / "tokenizer_experiments_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    print(f"Wrote report to {report_path}")
    return report


def encode_file_to_uint16_npy(
    tokenizer: Tokenizer,
    input_path: Path,
    output_path: Path,
    chunk_bytes: int,
    write_buffer_tokens: int,
    overwrite: bool,
    show_progress: bool,
) -> dict[str, int | str]:
    """把文本文件编码成 uint16 NumPy 数组；先写 raw 临时文件，再生成带 shape 的 .npy。"""
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp_uint16")
    total_tokens = 0
    max_token_id = 0
    token_buffer: list[int] = []

    def flush_buffer() -> None:
        nonlocal total_tokens, max_token_id, token_buffer
        if not token_buffer:
            return
        local_max = max(token_buffer)
        if local_max > UINT16_MAX:
            raise ValueError(f"Token id {local_max} exceeds uint16 maximum {UINT16_MAX}")
        np.asarray(token_buffer, dtype=np.uint16).tofile(tmp_file)
        total_tokens += len(token_buffer)
        max_token_id = max(max_token_id, local_max)
        token_buffer = []

    with tmp_path.open("wb") as tmp_file:
        text_chunks = iter_utf8_text_chunks(
            input_path=input_path,
            chunk_bytes=chunk_bytes,
            show_progress=show_progress,
            description=f"encode {input_path.name}",
        )
        for token_id in tokenizer.encode_iterable(text_chunks):
            token_buffer.append(token_id)
            if len(token_buffer) >= write_buffer_tokens:
                flush_buffer()
        flush_buffer()

    token_array = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(total_tokens,))
    offset = 0
    tmp_size = tmp_path.stat().st_size
    with tqdm(
        total=tmp_size,
        desc=f"write {output_path.name}",
        unit="B",
        unit_scale=True,
        disable=not show_progress,
    ) as progress:
        with tmp_path.open("rb") as tmp_file:
            while raw_chunk := tmp_file.read(write_buffer_tokens * np.dtype(np.uint16).itemsize):
                chunk_array = np.frombuffer(raw_chunk, dtype=np.uint16)
                token_array[offset : offset + len(chunk_array)] = chunk_array
                offset += len(chunk_array)
                progress.update(len(raw_chunk))
    token_array.flush()
    tmp_path.unlink()

    return {
        "input_path": str(input_path),
        "output_path": str(output_path),
        "tokens": total_tokens,
        "max_token_id": max_token_id,
        "dtype": "uint16",
    }


def run_encode_datasets(args: argparse.Namespace) -> dict[str, Any]:
    tiny_tokenizer = load_tokenizer("tinystories_10k", args.tinystories_artifacts)
    owt_tokenizer = load_tokenizer("owt_32k", args.owt_artifacts)

    jobs = [
        (
            tiny_tokenizer.tokenizer,
            args.tinystories_train,
            args.output_dir / "tinystories_train_uint16.npy",
        ),
        (
            tiny_tokenizer.tokenizer,
            args.tinystories_dev,
            args.output_dir / "tinystories_dev_uint16.npy",
        ),
        (
            owt_tokenizer.tokenizer,
            args.owt_train,
            args.output_dir / "owt_train_uint16.npy",
        ),
        (
            owt_tokenizer.tokenizer,
            args.owt_dev,
            args.output_dir / "owt_dev_uint16.npy",
        ),
    ]

    results = []
    for tokenizer, input_path, output_path in jobs:
        start = time.perf_counter()
        result = encode_file_to_uint16_npy(
            tokenizer=tokenizer,
            input_path=input_path,
            output_path=output_path,
            chunk_bytes=args.chunk_bytes,
            write_buffer_tokens=args.write_buffer_tokens,
            overwrite=args.overwrite,
            show_progress=not args.no_progress,
        )
        result["seconds"] = time.perf_counter() - start
        results.append(result)
        print(json.dumps(result, indent=2))

    summary = {"encoded_datasets": results}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / "tokenized_datasets_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"Wrote summary to {summary_path}")
    return summary


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run tokenizer experiments for CS336 assignment 1.")
    parser.add_argument("--mode", choices=["report", "encode-datasets"], default="report")
    parser.add_argument("--tinystories-artifacts", type=Path, default=Path("artifacts/tinystories_bpe"))
    parser.add_argument("--owt-artifacts", type=Path, default=Path("artifacts/owt_bpe"))
    parser.add_argument("--tinystories-train", type=Path, default=Path("data/TinyStoriesV2-GPT4-train.txt"))
    parser.add_argument("--tinystories-dev", type=Path, default=Path("data/TinyStoriesV2-GPT4-valid.txt"))
    parser.add_argument("--owt-train", type=Path, default=Path("data/owt_train.txt"))
    parser.add_argument("--owt-dev", type=Path, default=Path("data/owt_valid.txt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/tokenizer_experiments"))
    parser.add_argument("--sample-documents", type=int, default=DEFAULT_SAMPLE_DOCUMENTS)
    parser.add_argument("--benchmark-bytes", type=int, default=DEFAULT_BENCHMARK_BYTES)
    parser.add_argument(
        "--chunk-bytes",
        "--chunk-chars",
        dest="chunk_bytes",
        type=int,
        default=DEFAULT_CHUNK_BYTES,
        help="Bytes to read per input chunk. --chunk-chars is kept as a backward-compatible alias.",
    )
    parser.add_argument("--write-buffer-tokens", type=int, default=DEFAULT_WRITE_BUFFER_TOKENS)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress bars.")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if args.mode == "report":
        run_report(args)
    else:
        run_encode_datasets(args)


if __name__ == "__main__":
    main()
