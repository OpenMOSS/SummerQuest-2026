# 大文件编码入口。切块只能卡 special token，随便腰斩 UTF-8 会把结果切坏。

from __future__ import annotations

import argparse
import multiprocessing
import os
import sys
import tempfile
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path

import numpy as np


SUBMISSION_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SUBMISSION_ROOT))

from cs336_basics.tokenizer import (  # noqa: E402
    Tokenizer,
    _find_chunk_boundaries,
    _read_utf8_range,
)


# worker 启动时读一次 tokenizer，后面一直蹭这个全局，省得每块都重新读 JSON。
_WORKER_TOKENIZER: Tokenizer | None = None
_WORKER_DTYPE: np.dtype | None = None


def _initialise_encoder_worker(
    vocab_path: str,
    merges_path: str,
    special_tokens: tuple[str, ...],
    dtype_name: str,
) -> None:
    global _WORKER_TOKENIZER, _WORKER_DTYPE
    _WORKER_TOKENIZER = Tokenizer.from_files(vocab_path, merges_path, list(special_tokens))
    _WORKER_DTYPE = np.dtype(dtype_name)


def _encode_chunk_to_file(task: tuple[int, str, int, int, str]) -> tuple[int, int, int]:
    if _WORKER_TOKENIZER is None or _WORKER_DTYPE is None:
        raise RuntimeError("encoding worker was not initialised")
    chunk_i, input_path, start, end, part_path = task
    text = _read_utf8_range(input_path, start, end, range_label="encoding")
    # fromiter 直接吃流，别先 list，一大文件真会把内存吃空。
    token_ids = np.fromiter(_WORKER_TOKENIZER.encode_iterable([text]), dtype=_WORKER_DTYPE)
    np.save(part_path, token_ids)
    return chunk_i, len(token_ids), end - start


def _print_progress(
    label: str,
    completed: int,
    total: int,
    processed: int,
    total_amount: int,
    *,
    unit: str,
) -> None:
    done_rate = completed / total if total else 1.0
    width = 28
    filled = min(width, int(width * done_rate))
    bar = "#" * filled + "-" * (width - filled)
    if unit == "MiB":
        rendered_progress = f"{processed / 2**20:.1f}/{total_amount / 2**20:.1f} MiB"
    else:
        rendered_progress = f"{processed:,}/{total_amount:,} {unit}"
    ending = "\n" if completed >= total else ""
    print(
        f"\r[{label}] [{bar}] {completed}/{total} ({done_rate:6.2%}) {rendered_progress}",
        end=ending,
        file=sys.stderr,
        flush=True,
    )


def _encode_serial_stream(
    input_path: Path,
    tokenizer: Tokenizer,
    dtype: np.dtype,
    temporary_output: Path,
    show_progress: bool,
) -> int:
    if show_progress:
        print("[encode] no safe special-token boundary; using one serial stream", file=sys.stderr)
    with input_path.open(encoding="utf-8") as input_file:
        token_ids = np.fromiter(tokenizer.encode_iterable(input_file), dtype=dtype)
    np.save(temporary_output, token_ids)
    if show_progress:
        _print_progress("encode", 1, 1, input_path.stat().st_size, input_path.stat().st_size, unit="MiB")
    return len(token_ids)


def _encode_safe_chunks(
    input_path: Path,
    vocab_path: Path,
    merges_path: Path,
    special_tokens: list[str],
    dtype: np.dtype,
    output_path: Path,
    num_workers: int,
    chunks_per_worker: int,
    show_progress: bool,
) -> tuple[int, int]:
    if num_workers < 1:
        raise ValueError("num_workers must be at least 1")
    if chunks_per_worker < 1:
        raise ValueError("chunks_per_worker must be at least 1")

    cut_list = _find_chunk_boundaries(
        input_path,
        desired_num_chunks=num_workers * chunks_per_worker,
        special_tokens=special_tokens,
    )
    raw_jobs = [
        (start, end) for start, end in zip(cut_list[:-1], cut_list[1:]) if end > start
    ]
    # 切不出安全边界就认命单线程，硬切只会快得不对。
    if len(raw_jobs) <= 1:
        tokenizer = Tokenizer.from_files(vocab_path, merges_path, special_tokens)
        with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.parts-", dir=output_path.parent) as temp_dir:
            temporary_output = Path(temp_dir) / "encoded.npy"
            token_count = _encode_serial_stream(
                input_path, tokenizer, dtype, temporary_output, show_progress
            )
            os.replace(temporary_output, output_path)
        return token_count, 1

    real_workers = min(num_workers, len(raw_jobs))
    with tempfile.TemporaryDirectory(prefix=f".{output_path.stem}.parts-", dir=output_path.parent) as temp_dir:
        temporary_root = Path(temp_dir)
        tasks = [
            (index, str(input_path), start, end, str(temporary_root / f"part-{index:05d}.npy"))
            for index, (start, end) in enumerate(raw_jobs)
        ]
        token_counts = [0] * len(tasks)
        completed_bytes = 0

        if real_workers == 1:
            _initialise_encoder_worker(str(vocab_path), str(merges_path), tuple(special_tokens), dtype.name)
            for completed, task in enumerate(tasks, start=1):
                chunk_index, token_count, source_bytes = _encode_chunk_to_file(task)
                token_counts[chunk_index] = token_count
                completed_bytes += source_bytes
                if show_progress:
                    _print_progress(
                        "encode", completed, len(tasks), completed_bytes, input_path.stat().st_size, unit="MiB"
                    )
        else:
            # 大数组让 worker 自己落盘，只回三个数，不然进程管道扛不住。
            with ProcessPoolExecutor(
                max_workers=real_workers,
                initializer=_initialise_encoder_worker,
                initargs=(str(vocab_path), str(merges_path), tuple(special_tokens), dtype.name),
            ) as pool:
                futures = {pool.submit(_encode_chunk_to_file, task): task[0] for task in tasks}
                completed = 0
                while futures:
                    done, _ = wait(futures, return_when=FIRST_COMPLETED)
                    for future in done:
                        futures.pop(future)
                        chunk_index, token_count, source_bytes = future.result()
                        token_counts[chunk_index] = token_count
                        completed += 1
                        completed_bytes += source_bytes
                        if show_progress:
                            _print_progress(
                                "encode",
                                completed,
                                len(tasks),
                                completed_bytes,
                                input_path.stat().st_size,
                                unit="MiB",
                            )

        total_tokens = sum(token_counts)
        temporary_output = temporary_root / "encoded.npy"
        merged = np.lib.format.open_memmap(
            temporary_output,
            mode="w+",
            dtype=dtype,
            shape=(total_tokens,),
        )
        offset = 0
        for index, token_count in enumerate(token_counts):
            part = np.load(temporary_root / f"part-{index:05d}.npy", mmap_mode="r")
            merged[offset : offset + token_count] = part
            offset += token_count
            if show_progress:
                _print_progress("merge", index + 1, len(tasks), offset, total_tokens, unit="tokens")
            del part
        # 先关 memmap 再 replace，Windows 对没关的文件特别较真。
        del merged
        os.replace(temporary_output, output_path)
    return total_tokens, real_workers


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--vocab", type=Path, required=True)
    parser.add_argument("--merges", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--special-token", action="append", default=[])
    parser.add_argument("--dtype", choices=("uint16", "uint32"), default="uint16")
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--chunks-per-worker", type=int, default=4)
    parser.add_argument("--quiet", action="store_true", help="关闭编码和合并进度条")
    args = parser.parse_args()
    if not args.input.is_file():
        parser.error(f"input file does not exist or is not a regular file: {args.input}")
    if not args.vocab.is_file() or not args.merges.is_file():
        parser.error("--vocab and --merges must be existing regular files")
    if args.num_workers < 1:
        parser.error("--num-workers must be at least 1")
    if args.chunks_per_worker < 1:
        parser.error("--chunks-per-worker must be at least 1")

    tokenizer = Tokenizer.from_files(args.vocab, args.merges, args.special_token)
    dtype = np.dtype(args.dtype)
    # vocab 塞不进 uint16 就直接喊停，静默截断最坑人。
    if dtype == np.dtype("uint16") and max(tokenizer.vocab) > np.iinfo(np.uint16).max:
        raise ValueError("the tokenizer vocabulary does not fit in uint16")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    token_count, effective_workers = _encode_safe_chunks(
        args.input,
        args.vocab,
        args.merges,
        args.special_token,
        dtype,
        args.output,
        args.num_workers,
        args.chunks_per_worker,
        show_progress=not args.quiet,
    )
    print(
        f"encoded {token_count} tokens to {args.output} using {effective_workers} worker(s)",
        flush=True,
    )


if __name__ == "__main__":
    # Windows spawn 老规矩，别删。
    multiprocessing.freeze_support()
    main()
