"""并行编码数据集，并将 token ID 保存为 uint16 NumPy 数组。"""

from __future__ import annotations

import os
import pickle
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from pathlib import Path
from queue import Empty
from multiprocessing import Manager

import numpy as np
from tqdm import tqdm

from cs336_basics.tokenizer import Tokenizer, _find_chunk_boundaries


SPECIAL_TOKEN = "<|endoftext|>"
OUTPUT_DIRECTORY = Path("artifacts/token_ids")
TOKENIZER_FILES = {
    # "tinystories": Path("artifacts/tinystories_bpe_10k.pkl"),
    "owt": Path("artifacts/owt_bpe_32k.pkl"),
}
DATASETS = {
    # "tinystories": {
    #     "train": Path("data/TinyStoriesV2-GPT4-train.txt"),
    #     "valid": Path("data/TinyStoriesV2-GPT4-valid.txt"),
    # },
    "owt": {
        "train": Path("data/owt_train.txt"),
        "valid": Path("data/owt_valid.txt"),
    },
}
NUM_PROCESSES = min(32, os.cpu_count() or 1)
WRITE_BUFFER_SIZE = 1_000_000
READ_BUFFER_SIZE = 16 << 20
PROGRESS_UPDATE_BYTES = 32 << 20


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """加载训练脚本保存的词表和 merge 规则。"""
    with tokenizer_path.open("rb") as tokenizer_file:
        serialized = pickle.load(tokenizer_file)
    return Tokenizer(
        serialized["vocab"],
        serialized["merges"],
        serialized.get("special_tokens", [SPECIAL_TOKEN]),
    )


def iter_chunk_token_ids(
    tokenizer_path: str,
    input_path: str,
    start: int,
    end: int,
    progress_queue,
    chunk_index: int,
):
    """流式编码安全文本分块，并在每个文档边界上报完成的字节数。"""
    tokenizer = load_tokenizer(Path(tokenizer_path))
    special_token_bytes = SPECIAL_TOKEN.encode("utf-8")
    unreported_bytes = 0
    with open(input_path, "rb") as input_file:
        input_file.seek(start)
        remaining_bytes = end - start
        buffer = b""
        while remaining_bytes:
            block = input_file.read(min(READ_BUFFER_SIZE, remaining_bytes))
            remaining_bytes -= len(block)
            buffer += block
            last_boundary = buffer.rfind(special_token_bytes)
            if last_boundary < 0:
                continue
            split_at = last_boundary + len(special_token_bytes)
            text_bytes, buffer = buffer[:split_at], buffer[split_at:]
            yield from tokenizer.iter_encode(text_bytes.decode("utf-8"))
            unreported_bytes += len(text_bytes)
            if unreported_bytes >= PROGRESS_UPDATE_BYTES:
                progress_queue.put((chunk_index, unreported_bytes))
                unreported_bytes = 0
        if buffer:
            yield from tokenizer.iter_encode(buffer.decode("utf-8"))
            unreported_bytes += len(buffer)
    if unreported_bytes:
        progress_queue.put((chunk_index, unreported_bytes))


def count_chunk(task) -> int:
    """子进程任务：统计一个分块的 token 数。"""
    tokenizer_path, input_path, start, end, progress_queue, chunk_index = task
    return sum(1 for _ in iter_chunk_token_ids(tokenizer_path, input_path, start, end, progress_queue, chunk_index))


def write_chunk(task) -> int:
    """子进程任务：把一个分块写入最终数组的专属、不重叠区间。"""
    tokenizer_path, input_path, start, end, output_path, output_offset, expected_count, progress_queue, chunk_index = task
    output_array = np.load(output_path, mmap_mode="r+")
    write_position = output_offset
    buffer: list[int] = []
    for token_id in iter_chunk_token_ids(tokenizer_path, input_path, start, end, progress_queue, chunk_index):
        buffer.append(token_id)
        if len(buffer) == WRITE_BUFFER_SIZE:
            output_array[write_position : write_position + len(buffer)] = buffer
            write_position += len(buffer)
            buffer.clear()
    if buffer:
        output_array[write_position : write_position + len(buffer)] = buffer
        write_position += len(buffer)
    output_array.flush()

    written_count = write_position - output_offset
    if written_count != expected_count:
        raise RuntimeError(f"分块 token 数改变：预计 {expected_count:,}，实际 {written_count:,}")
    return written_count


def run_parallel_tasks(function, tasks: list[tuple], stage: str, total_bytes: int) -> list[int]:
    """运行分块任务，并根据 worker 上报的实际字节数显示实时进度。"""
    results = [0] * len(tasks)
    with Manager() as manager, ProcessPoolExecutor(max_workers=min(NUM_PROCESSES, len(tasks))) as executor:
        progress_queue = manager.Queue()
        future_to_index = {
            executor.submit(function, (*task, progress_queue, index)): index for index, task in enumerate(tasks)
        }
        pending_futures = set(future_to_index)
        with tqdm(total=total_bytes, desc=stage, unit="B", unit_scale=True, dynamic_ncols=True) as progress_bar:
            while pending_futures:
                completed, pending_futures = wait(pending_futures, timeout=0.2, return_when=FIRST_COMPLETED)
                while True:
                    try:
                        _, completed_bytes = progress_queue.get_nowait()
                    except Empty:
                        break
                    progress_bar.update(completed_bytes)
                for future in completed:
                    results[future_to_index[future]] = future.result()
    return results


def encode_to_uint16(tokenizer_path: Path, input_path: Path, output_path: Path) -> None:
    """按特殊 token 分块后两遍并行编码，顺序写入单个 uint16 .npy 文件。"""
    print(f"\n开始处理：{input_path}（{NUM_PROCESSES} 个进程）")
    boundaries = _find_chunk_boundaries(input_path, NUM_PROCESSES, SPECIAL_TOKEN.encode("utf-8"))
    chunks = [(start, end) for start, end in zip(boundaries, boundaries[1:]) if start != end]
    common_task_prefix = (str(tokenizer_path), str(input_path))

    count_tasks = [(*common_task_prefix, start, end) for start, end in chunks]
    total_bytes = sum(end - start for start, end in chunks)
    chunk_token_counts = run_parallel_tasks(count_chunk, count_tasks, "计数", total_bytes)
    token_count = sum(chunk_token_counts)
    print(f"  token 总数：{token_count:,}；开始并行写入 {output_path}", flush=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_array = np.lib.format.open_memmap(output_path, mode="w+", dtype=np.uint16, shape=(token_count,))
    output_array.flush()
    del output_array

    offsets = np.cumsum([0, *chunk_token_counts[:-1]], dtype=np.int64)
    write_tasks = [
        (*common_task_prefix, start, end, str(output_path), int(offset), token_count_for_chunk)
        for (start, end), offset, token_count_for_chunk in zip(chunks, offsets, chunk_token_counts)
    ]
    run_parallel_tasks(write_chunk, write_tasks, "写入", total_bytes)
    print(f"  完成：{output_path}（{output_path.stat().st_size / 2**30:.2f} GiB）", flush=True)


def main() -> None:
    """编码配置中每个数据集的 train/valid split。"""
    for dataset_name, splits in DATASETS.items():
        tokenizer_path = TOKENIZER_FILES[dataset_name]
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"找不到 {dataset_name} tokenizer：{tokenizer_path}")
        tokenizer = load_tokenizer(tokenizer_path)
        if max(tokenizer.vocab) > np.iinfo(np.uint16).max:
            raise ValueError(f"{dataset_name} 的词表 ID 超出 uint16 范围")

        for split_name, input_path in splits.items():
            if not input_path.is_file():
                raise FileNotFoundError(f"找不到数据文件：{input_path}")
            output_path = OUTPUT_DIRECTORY / f"{dataset_name}_{split_name}_uint16.npy"
            encode_to_uint16(tokenizer_path, input_path, output_path)


if __name__ == "__main__":
    main()
