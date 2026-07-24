"""运行题目要求的双 tokenizer 对照实验。

脚本只输出聚合统计，不保存抽样文档正文，便于后续生成公开、脱敏的实验报告。
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections.abc import Iterator, Sequence
from pathlib import Path

from cs336_basics.tokenizer import BPETokenizer


SPECIAL_TOKEN = "<|endoftext|>"
PILE_BYTES = 825_000_000_000


def iter_documents(path: Path, chunk_size: int = 1024 * 1024) -> Iterator[str]:
    """按特殊 token 流式切分文档，避免一次性读取整个验证集。"""
    pending = ""
    with path.open("r", encoding="utf-8") as input_file:
        while chunk := input_file.read(chunk_size):
            pending += chunk
            parts = pending.split(SPECIAL_TOKEN)
            for document in parts[:-1]:
                if document.strip():
                    yield document + SPECIAL_TOKEN
            pending = parts[-1]

    if pending.strip():
        yield pending + SPECIAL_TOKEN


def reservoir_sample_documents(path: Path, sample_size: int, seed: int) -> tuple[list[str], int]:
    """使用蓄水池抽样，在不知道文档总数时等概率抽取文档。"""
    random_generator = random.Random(seed)
    sample: list[str] = []
    document_count = 0

    for document_count, document in enumerate(iter_documents(path), start=1):
        if document_count <= sample_size:
            sample.append(document)
            continue
        replacement_index = random_generator.randrange(document_count)
        if replacement_index < sample_size:
            sample[replacement_index] = document

    if len(sample) != sample_size:
        raise ValueError(f"{path.name} 只有 {len(sample)} 篇非空文档，无法抽取 {sample_size} 篇")
    return sample, document_count


def encode_statistics(tokenizer: BPETokenizer, documents: Sequence[str]) -> dict[str, float | int]:
    """计算一组文档的字节数、token 数和压缩率。"""
    byte_count = sum(len(document.encode("utf-8")) for document in documents)
    token_count = sum(len(tokenizer.encode(document)) for document in documents)
    if token_count == 0:
        raise ValueError("抽样文档编码后没有 token")
    return {
        "byte_count": byte_count,
        "token_count": token_count,
        "bytes_per_token": byte_count / token_count,
    }


def benchmark_encoding(
    tokenizer: BPETokenizer,
    documents: Sequence[str],
    minimum_seconds: float = 1.0,
    maximum_repeats: int = 64,
) -> dict[str, float | int]:
    """重复编码抽样文档，降低极短计时带来的误差。"""
    bytes_per_repeat = sum(len(document.encode("utf-8")) for document in documents)
    repeats = 0
    total_tokens = 0
    start_time = time.perf_counter()

    while repeats < maximum_repeats:
        for document in documents:
            total_tokens += len(tokenizer.encode(document))
        repeats += 1
        elapsed_seconds = time.perf_counter() - start_time
        if elapsed_seconds >= minimum_seconds:
            break

    elapsed_seconds = time.perf_counter() - start_time
    total_bytes = bytes_per_repeat * repeats
    return {
        "repeats": repeats,
        "elapsed_seconds": elapsed_seconds,
        "byte_count": total_bytes,
        "token_count": total_tokens,
        "bytes_per_second": total_bytes / max(elapsed_seconds, 1e-12),
    }


def load_full_corpus_throughput(stats_path: Path) -> float:
    """读取完整训练集编码阶段记录的实测吞吐。"""
    payload = json.loads(stats_path.read_text(encoding="utf-8"))
    train_stats = payload.get("train")
    if not isinstance(train_stats, dict):
        raise ValueError(f"{stats_path.name} 缺少 train 统计")
    bytes_per_second = train_stats.get("bytes_per_second")
    if not isinstance(bytes_per_second, (int, float)) or bytes_per_second <= 0:
        raise ValueError(f"{stats_path.name} 的 bytes_per_second 无效")
    return float(bytes_per_second)


def pile_time_estimate(bytes_per_second: float) -> dict[str, float]:
    """根据实测吞吐估算编码 825GB 文本的时间。"""
    seconds = PILE_BYTES / bytes_per_second
    return {
        "dataset_bytes": PILE_BYTES,
        "seconds": seconds,
        "hours": seconds / 3600,
        "days": seconds / 86400,
    }


def parse_args() -> argparse.Namespace:
    """解析实验所需的公开相对路径。"""
    parser = argparse.ArgumentParser(description="运行 CS336 双 tokenizer 对照实验")
    parser.add_argument("--tinystories-text", type=Path, required=True)
    parser.add_argument("--owt-text", type=Path, required=True)
    parser.add_argument("--tinystories-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--owt-tokenizer-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sample-size", type=int, default=10)
    parser.add_argument("--seed", type=int, default=20260714)
    return parser.parse_args()


def load_tokenizer(directory: Path) -> BPETokenizer:
    """从标准 vocab/merges 文件加载 tokenizer。"""
    return BPETokenizer.from_files(
        str(directory / "vocab.json"),
        str(directory / "merges.json"),
        special_tokens=[SPECIAL_TOKEN],
    )


def main() -> None:
    """抽样、编码、计时并写出结构化结果。"""
    args = parse_args()
    if args.sample_size <= 0:
        raise ValueError("sample_size 必须为正数")

    tinystories_documents, tinystories_document_count = reservoir_sample_documents(
        args.tinystories_text,
        args.sample_size,
        args.seed,
    )
    owt_documents, owt_document_count = reservoir_sample_documents(
        args.owt_text,
        args.sample_size,
        args.seed,
    )

    tinystories_tokenizer = load_tokenizer(args.tinystories_tokenizer_dir)
    owt_tokenizer = load_tokenizer(args.owt_tokenizer_dir)

    tinystories_full_throughput = load_full_corpus_throughput(
        args.tinystories_tokenizer_dir / "tokenizer_stats.json"
    )
    owt_full_throughput = load_full_corpus_throughput(args.owt_tokenizer_dir / "tokenizer_stats.json")

    result = {
        "sample": {
            "source_split": "validation",
            "sample_size_per_dataset": args.sample_size,
            "seed": args.seed,
            "tinystories_document_count": tinystories_document_count,
            "owt_document_count": owt_document_count,
            "document_text_saved": False,
        },
        "compression": {
            "tinystories_with_tinystories_tokenizer": encode_statistics(
                tinystories_tokenizer,
                tinystories_documents,
            ),
            "owt_with_owt_tokenizer": encode_statistics(owt_tokenizer, owt_documents),
            "owt_with_tinystories_tokenizer": encode_statistics(
                tinystories_tokenizer,
                owt_documents,
            ),
        },
        "sample_single_process_throughput": {
            "tinystories_tokenizer": benchmark_encoding(
                tinystories_tokenizer,
                tinystories_documents,
            ),
            "owt_tokenizer": benchmark_encoding(owt_tokenizer, owt_documents),
        },
        "full_corpus_measured_throughput": {
            "tinystories_bytes_per_second": tinystories_full_throughput,
            "owt_bytes_per_second": owt_full_throughput,
        },
        "pile_825gb_estimate": {
            "using_tinystories_full_corpus_throughput": pile_time_estimate(
                tinystories_full_throughput
            ),
            "using_owt_full_corpus_throughput": pile_time_estimate(owt_full_throughput),
        },
        "uint16_explanation": (
            "两套词表最多分别为 10000 和 32000，token ID 均小于 uint16 可表示的 65536；"
            "因此每个 token 只需 2 字节，比 int32 节省一半存储。"
        ),
    }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"结果已写入: {args.output}")


if __name__ == "__main__":
    main()
