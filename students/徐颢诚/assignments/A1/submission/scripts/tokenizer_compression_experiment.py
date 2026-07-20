"""抽样 TinyStories 和 OpenWebText 文档，测量 BPE tokenizer 的压缩率。"""

from __future__ import annotations

import pickle
import random
from collections.abc import Iterator
from pathlib import Path

from cs336_basics.tokenizer import Tokenizer


DATASETS = {
    "TinyStories": Path("data/TinyStoriesV2-GPT4-valid.txt"),
    "OpenWebText": Path("data/owt_valid.txt"),
}
TOKENIZER_FILES = {
    "TinyStories-10K": Path("artifacts/tinystories_bpe_10k.pkl"),
    "OpenWebText-32K": Path("artifacts/owt_bpe_32k.pkl"),
}
SPECIAL_TOKEN = "<|endoftext|>"
SAMPLE_SIZE = 10
RANDOM_SEED = 336
OUTPUT_PATH = Path("artifacts/tokenizer_compression_samples.pkl")


def iter_documents(input_path: Path) -> Iterator[str]:
    """流式读取由 <|endoftext|> 分隔的文档，不将整个文件载入内存。"""
    remainder = ""
    with input_path.open(encoding="utf-8") as input_file:
        while chunk := input_file.read(1 << 20):
            parts = (remainder + chunk).split(SPECIAL_TOKEN)
            remainder = parts.pop()
            yield from (document for document in parts if document)
    if remainder:
        yield remainder


def reservoir_sample_documents(input_path: Path, sample_size: int, seed: int) -> list[str]:
    """从未知文档总数的文件中均匀抽取固定数量的文档。"""
    random_generator = random.Random(seed)
    sample: list[str] = []
    for index, document in enumerate(iter_documents(input_path)):
        if index < sample_size:
            sample.append(document)
        else:
            replacement_index = random_generator.randint(0, index)
            if replacement_index < sample_size:
                sample[replacement_index] = document
    if len(sample) < sample_size:
        raise ValueError(f"{input_path} 中只有 {len(sample)} 篇非空文档，无法抽取 {sample_size} 篇")
    return sample


def load_tokenizer(tokenizer_path: Path) -> Tokenizer:
    """从训练脚本保存的 pickle 文件恢复 Tokenizer。"""
    with tokenizer_path.open("rb") as tokenizer_file:
        serialized = pickle.load(tokenizer_file)
    return Tokenizer(
        serialized["vocab"],
        serialized["merges"],
        serialized.get("special_tokens", [SPECIAL_TOKEN]),
    )


def encode_and_measure(tokenizer: Tokenizer, documents: list[str]) -> tuple[list[list[int]], int, int, float]:
    """编码文档并返回 IDs、总字节数、总 token 数和 bytes/token 压缩率。"""
    encoded_documents = [tokenizer.encode(document) for document in documents]
    byte_count = sum(len(document.encode("utf-8")) for document in documents)
    token_count = sum(len(ids) for ids in encoded_documents)
    if token_count == 0:
        raise ValueError("抽样文档编码后没有 token，无法计算压缩率")
    return encoded_documents, byte_count, token_count, byte_count / token_count


def main() -> None:
    """运行抽样和压缩率实验，并将编码结果保存到磁盘。"""
    tokenizers = {}
    for name, tokenizer_path in TOKENIZER_FILES.items():
        if not tokenizer_path.is_file():
            raise FileNotFoundError(f"找不到已训练 tokenizer：{tokenizer_path}")
        tokenizers[name] = load_tokenizer(tokenizer_path)

    samples = {}
    for offset, (name, dataset_path) in enumerate(DATASETS.items()):
        if not dataset_path.is_file():
            raise FileNotFoundError(f"找不到数据集：{dataset_path}")
        samples[name] = reservoir_sample_documents(dataset_path, SAMPLE_SIZE, RANDOM_SEED + offset)
        print(f"已从 {name} 抽取 {len(samples[name])} 篇文档。")

    results: dict[str, dict[str, object]] = {}
    print("\n压缩率（原始 UTF-8 bytes / token；越高表示每个 token 覆盖的文本越多）：")
    for dataset_name, documents in samples.items():
        for tokenizer_name, tokenizer in tokenizers.items():
            encoded_documents, byte_count, token_count, ratio = encode_and_measure(tokenizer, documents)
            result_name = f"{dataset_name} × {tokenizer_name}"
            results[result_name] = {
                "byte_count": byte_count,
                "token_count": token_count,
                "bytes_per_token": ratio,
                "encoded_documents": encoded_documents,
            }
            print(f"{result_name}: {ratio:.4f} bytes/token ({byte_count:,} bytes, {token_count:,} tokens)")

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_PATH.open("wb") as output_file:
        pickle.dump(
            {
                "sample_size": SAMPLE_SIZE,
                "random_seed": RANDOM_SEED,
                "samples": samples,
                "results": results,
            },
            output_file,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"\n抽样文档和 token IDs 已保存到：{OUTPUT_PATH}")


if __name__ == "__main__":
    main()
