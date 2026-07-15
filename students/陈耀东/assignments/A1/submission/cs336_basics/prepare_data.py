"""训练 BPE tokenizer，并把文本编码为 uint16 token 数据。"""

from __future__ import annotations

import argparse
import cProfile
import io
import json
import os
import pstats
import shutil
import sys
import tempfile
import time
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from cs336_basics.bpe_train import find_chunk_boundaries, resolve_num_processes, train_bpe
from cs336_basics.tokenizer import BPETokenizer
from cs336_basics.tokenizer_io import save_tokenizer_files

_ENCODE_WORKER_TOKENIZER: BPETokenizer | None = None
_ENCODE_CACHE_CAPACITY = 131_072


def iter_text_lines(path: str | os.PathLike) -> Iterator[str]:
    """逐行读取文本，避免一次性把大语料全部载入内存。"""
    with Path(path).open("r", encoding="utf-8") as input_file:
        yield from input_file


def encode_text_to_uint16(
    tokenizer: BPETokenizer,
    input_path: str | os.PathLike,
    output_path: str | os.PathLike,
    flush_tokens: int = 1_000_000,
    num_processes: int | None = None,
) -> int:
    """按 special-token 安全边界并行编码，并按原顺序拼接 uint16 结果。"""
    if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("词表 ID 超出 uint16 可表示范围")

    input_path = Path(input_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    worker_count = resolve_num_processes(input_path, tokenizer.special_tokens, num_processes)
    if worker_count == 1:
        return encode_text_to_uint16_serial(
            tokenizer,
            input_path,
            output_path,
            flush_tokens=flush_tokens,
        )

    boundaries = find_chunk_boundaries(
        input_path,
        desired_num_chunks=worker_count * 4,
        split_special_tokens=tuple(token.encode("utf-8") for token in tokenizer.special_tokens),
    )
    ranges = [(start, end) for start, end in zip(boundaries[:-1], boundaries[1:]) if end > start]
    if len(ranges) <= 1:
        return encode_text_to_uint16_serial(
            tokenizer,
            input_path,
            output_path,
            flush_tokens=flush_tokens,
        )

    temporary_output = output_path.with_name(f".{output_path.name}.tmp")
    try:
        with tempfile.TemporaryDirectory(prefix=f".{output_path.name}.parts-", dir=output_path.parent) as part_dir:
            tasks = [
                (
                    str(input_path),
                    start,
                    end,
                    str(Path(part_dir) / f"{index:05d}.bin"),
                    flush_tokens,
                )
                for index, (start, end) in enumerate(ranges)
            ]
            with ProcessPoolExecutor(
                max_workers=min(worker_count, len(tasks)),
                initializer=initialize_encode_worker,
                initargs=(tokenizer.vocab, tokenizer.merges, tokenizer.special_tokens),
            ) as executor:
                token_counts = list(executor.map(encode_chunk_to_uint16, tasks, chunksize=1))

            with temporary_output.open("wb") as combined_output:
                for _, _, _, part_path, _ in tasks:
                    with Path(part_path).open("rb") as part_file:
                        shutil.copyfileobj(part_file, combined_output, length=16 * 1024 * 1024)
        temporary_output.replace(output_path)
    finally:
        temporary_output.unlink(missing_ok=True)
    return sum(token_counts)


def encode_text_to_uint16_serial(
    tokenizer: BPETokenizer,
    input_path: Path,
    output_path: Path,
    flush_tokens: int,
) -> int:
    """保留低开销串行路径，适合小文件和正确性对照。"""
    token_buffer: list[int] = []
    token_count = 0

    with output_path.open("wb") as output_file:
        for token_id in tokenizer.encode_iterable(iter_text_lines(input_path)):
            token_buffer.append(token_id)
            if len(token_buffer) >= flush_tokens:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                token_count += len(token_buffer)
                token_buffer.clear()

        if token_buffer:
            np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
            token_count += len(token_buffer)

    return token_count


def initialize_encode_worker(
    vocab: dict[int, bytes],
    merges: list[tuple[bytes, bytes]],
    special_tokens: list[str],
) -> None:
    """为每个编码进程只构造一次 tokenizer，并开启有上限的 pre-token 缓存。"""
    global _ENCODE_WORKER_TOKENIZER
    _ENCODE_WORKER_TOKENIZER = BPETokenizer(
        vocab=vocab,
        merges=merges,
        special_tokens=special_tokens,
        cache_capacity=_ENCODE_CACHE_CAPACITY,
    )


def encode_chunk_to_uint16(task: tuple[str, int, int, str, int]) -> int:
    """编码一个安全文件块到独立 part 文件，并返回 token 数。"""
    input_path, start, end, output_path, flush_tokens = task
    tokenizer = _ENCODE_WORKER_TOKENIZER
    if tokenizer is None:
        raise RuntimeError("编码 worker 尚未初始化 tokenizer")

    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        text = input_file.read(end - start).decode("utf-8")
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    token_buffer: list[int] = []
    token_count = 0
    with Path(output_path).open("wb") as output_file:
        # StringIO 逐行迭代，避免 encode 一次性创建整块 token id 列表。
        for token_id in tokenizer.encode_iterable(io.StringIO(text)):
            token_buffer.append(token_id)
            if len(token_buffer) >= flush_tokens:
                np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
                token_count += len(token_buffer)
                token_buffer.clear()
        if token_buffer:
            np.asarray(token_buffer, dtype=np.uint16).tofile(output_file)
            token_count += len(token_buffer)
    return token_count


def describe_token(token: bytes) -> dict[str, str | int]:
    """把 bytes token 转成可写入 JSON、也便于人工检查的描述。"""
    return {
        "hex": token.hex(),
        "utf8_with_replacement": token.decode("utf-8", errors="replace"),
        "byte_length": len(token),
    }


def peak_process_rss_bytes() -> int | None:
    """在支持 getrusage 的平台返回当前进程历史峰值 RSS。"""
    try:
        import resource

        getrusage = getattr(resource, "getrusage", None)
        rusage_self = getattr(resource, "RUSAGE_SELF", None)
        if not callable(getrusage) or rusage_self is None:
            return None
        peak_rss = getattr(getrusage(rusage_self), "ru_maxrss", None)
    except (AttributeError, ImportError, OSError):
        return None
    if not isinstance(peak_rss, (int, float)):
        return None

    # Linux 以 KiB 返回，macOS 以 bytes 返回。
    return int(peak_rss if sys.platform == "darwin" else peak_rss * 1024)


def write_profile(profile: cProfile.Profile, output_path: Path) -> None:
    """写出已去除目录前缀的累计耗时热点，避免泄露内部绝对路径。"""
    stream = io.StringIO()
    stats = pstats.Stats(profile, stream=stream)
    stats.strip_dirs().sort_stats("cumulative").print_stats(30)
    output_path.write_text(stream.getvalue(), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="准备 CS336 语言模型训练数据")
    parser.add_argument("--train-text", type=Path, required=True)
    parser.add_argument("--valid-text", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--vocab-size", type=int, default=10_000)
    parser.add_argument(
        "--num-processes",
        type=int,
        default=None,
        help="BPE 预分词进程数；大文件默认最多使用 16 个进程",
    )
    parser.add_argument(
        "--special-token",
        action="append",
        default=None,
        help="可重复传入；默认使用 <|endoftext|>",
    )
    parser.add_argument(
        "--profile-bpe",
        action="store_true",
        help="只 profile train_bpe，并在输出目录写入 bpe_profile.txt",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    special_tokens = args.special_token or ["<|endoftext|>"]
    args.output_dir.mkdir(parents=True, exist_ok=True)

    profiler = cProfile.Profile() if args.profile_bpe else None
    start_time = time.perf_counter()
    if profiler is not None:
        profiler.enable()
    vocab, merges = train_bpe(
        input_path=args.train_text,
        vocab_size=args.vocab_size,
        special_tokens=special_tokens,
        num_processes=args.num_processes,
    )
    if profiler is not None:
        profiler.disable()
    tokenizer_seconds = time.perf_counter() - start_time
    peak_rss_after_bpe = peak_process_rss_bytes()

    profile_path: Path | None = None
    if profiler is not None:
        profile_path = args.output_dir / "bpe_profile.txt"
        write_profile(profiler, profile_path)

    vocab_path = args.output_dir / "vocab.json"
    merges_path = args.output_dir / "merges.json"
    save_tokenizer_files(vocab, merges, vocab_path, merges_path)
    tokenizer = BPETokenizer(
        vocab,
        merges,
        special_tokens=special_tokens,
        cache_capacity=_ENCODE_CACHE_CAPACITY,
    )

    train_encode_start = time.perf_counter()
    train_tokens = encode_text_to_uint16(
        tokenizer,
        args.train_text,
        args.output_dir / "train.bin",
        num_processes=args.num_processes,
    )
    train_encode_seconds = time.perf_counter() - train_encode_start

    valid_encode_start = time.perf_counter()
    valid_tokens = encode_text_to_uint16(
        tokenizer,
        args.valid_text,
        args.output_dir / "valid.bin",
        num_processes=args.num_processes,
    )
    valid_encode_seconds = time.perf_counter() - valid_encode_start

    longest_token: bytes = max(vocab.values(), key=lambda token: len(token))
    special_token_bytes = {token.encode("utf-8") for token in special_tokens}
    ordinary_tokens = [token for token in vocab.values() if token not in special_token_bytes]
    longest_ordinary_token: bytes = max(ordinary_tokens, key=lambda token: len(token))
    train_bytes = args.train_text.stat().st_size
    valid_bytes = args.valid_text.stat().st_size
    if train_tokens == 0 or valid_tokens == 0:
        raise ValueError("训练集和验证集编码后都必须至少包含一个 token")

    stats = {
        "tokenizer_train_seconds": tokenizer_seconds,
        "peak_process_rss_after_bpe_bytes": peak_rss_after_bpe,
        "bpe_profile_file": profile_path.name if profile_path is not None else None,
        "actual_vocab_size": len(vocab),
        "bpe_num_processes": args.num_processes,
        "encoding_num_processes": args.num_processes,
        "special_tokens": special_tokens,
        "longest_token": describe_token(longest_token),
        "longest_ordinary_token": describe_token(longest_ordinary_token),
        "train": {
            "byte_count": train_bytes,
            "token_count": train_tokens,
            "encode_seconds": train_encode_seconds,
            "bytes_per_token": train_bytes / train_tokens,
            "bytes_per_second": train_bytes / max(train_encode_seconds, 1e-12),
        },
        "valid": {
            "byte_count": valid_bytes,
            "token_count": valid_tokens,
            "encode_seconds": valid_encode_seconds,
            "bytes_per_token": valid_bytes / valid_tokens,
            "bytes_per_second": valid_bytes / max(valid_encode_seconds, 1e-12),
        },
    }
    stats_path = args.output_dir / "tokenizer_stats.json"
    stats_path.write_text(
        json.dumps(stats, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"tokenizer 训练耗时: {tokenizer_seconds:.2f} 秒")
    print(f"实际词表大小: {len(vocab)}")
    print(f"最长 token: {longest_token!r}，长度 {len(longest_token)} bytes")
    print(f"最长普通 token: {longest_ordinary_token!r}，长度 {len(longest_ordinary_token)} bytes")
    print(
        f"训练集: {train_tokens} tokens，"
        f"{train_bytes / train_tokens:.3f} bytes/token，"
        f"{train_bytes / max(train_encode_seconds, 1e-12):.1f} bytes/s"
    )
    print(
        f"验证集: {valid_tokens} tokens，"
        f"{valid_bytes / valid_tokens:.3f} bytes/token，"
        f"{valid_bytes / max(valid_encode_seconds, 1e-12):.1f} bytes/s"
    )
    if peak_rss_after_bpe is not None:
        print(f"BPE 后进程峰值 RSS: {peak_rss_after_bpe / 1024**3:.3f} GiB")
    if profile_path is not None:
        print(f"BPE profile: {profile_path}")
    print(f"统计文件: {stats_path}")
    print(f"产物目录: {args.output_dir}")


if __name__ == "__main__":
    main()
