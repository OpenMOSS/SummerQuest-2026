import os
import time
import pickle
import numpy as np
from cs336_basics.tokenizer import Tokenizer, train_bpe

def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], filepath: str):
    with open(filepath, 'wb') as f:
        pickle.dump({'vocab': vocab, 'merges': merges}, f)

def load_tokenizer(filepath: str, special_tokens: list[str]) -> Tokenizer:
    with open(filepath, 'rb') as f:
        data = pickle.load(f)
    return Tokenizer(vocab=data['vocab'], merges=data['merges'], special_tokens=special_tokens)

def get_sample_documents(filepath: str, num_samples: int = 10) -> list[str]:
    if not os.path.exists(filepath):
        return [f"This is a sample document {i} for testing. " * 20 for i in range(num_samples)]
        
    with open(filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    docs = [doc.strip() for doc in content.split("<|endoftext|>") if doc.strip()]
    return docs[:num_samples]

def calculate_compression_ratio(tokenizer: Tokenizer, documents: list[str]) -> float:
    total_bytes = sum(len(doc.encode("utf-8")) for doc in documents)
    total_tokens = sum(len(tokenizer.encode(doc)) for doc in documents)
    return total_bytes / total_tokens if total_tokens > 0 else 0.0

def main():
    TS_DATA_PATH = "../data/TinyStoriesV2-GPT4-train.txt" 
    OWT_DATA_PATH = "../data/TinyStoriesV2-GPT4-valid.txt"
    
    TS_MODEL_PATH = "ts_tokenizer_10k.pkl"
    OWT_MODEL_PATH = "owt_tokenizer_32k.pkl"
    
    SPECIAL_TOKENS = ["<|endoftext|>"]
    
    if os.path.exists(TS_MODEL_PATH):
        ts_tokenizer = load_tokenizer(TS_MODEL_PATH, SPECIAL_TOKENS)
    else:
        ts_vocab, ts_merges = train_bpe(TS_DATA_PATH, vocab_size=10000, special_tokens=SPECIAL_TOKENS)
        save_tokenizer(ts_vocab, ts_merges, TS_MODEL_PATH)
        ts_tokenizer = Tokenizer(vocab=ts_vocab, merges=ts_merges, special_tokens=SPECIAL_TOKENS)

    if os.path.exists(OWT_MODEL_PATH):
        owt_tokenizer = load_tokenizer(OWT_MODEL_PATH, SPECIAL_TOKENS)
    else:
        owt_vocab, owt_merges = train_bpe(OWT_DATA_PATH, vocab_size=32000, special_tokens=SPECIAL_TOKENS)
        save_tokenizer(owt_vocab, owt_merges, OWT_MODEL_PATH)
        owt_tokenizer = Tokenizer(vocab=owt_vocab, merges=owt_merges, special_tokens=SPECIAL_TOKENS)


    ts_docs = get_sample_documents(TS_DATA_PATH, 10)
    owt_docs = get_sample_documents(OWT_DATA_PATH, 10)

    # ---------------------------------------------------------
    # 实验 (a): 计算各自分词器的压缩率
    # ---------------------------------------------------------
    print("\n=== 实验 (a): 压缩率计算 ===")
    ts_ratio = calculate_compression_ratio(ts_tokenizer, ts_docs)
    owt_ratio = calculate_compression_ratio(owt_tokenizer, owt_docs)
    print(f"TinyStories (10K) 压缩率: {ts_ratio:.2f} bytes/token")
    print(f"OpenWebText (32K) 压缩率: {owt_ratio:.2f} bytes/token")

    # ---------------------------------------------------------
    # 实验 (b): 跨领域分词测试
    # ---------------------------------------------------------
    print("\n=== 实验 (b): 跨领域分词 ===")
    cross_domain_ratio = calculate_compression_ratio(ts_tokenizer, owt_docs)
    print(f"使用 TinyStories(10K) 编码 OpenWebText 的压缩率: {cross_domain_ratio:.2f} bytes/token")

    # ---------------------------------------------------------
    # 实验 (c): 吞吐量估算
    # ---------------------------------------------------------
    print("\n=== 实验 (c): 吞吐量与耗时估算 ===")
    benchmark_text = "".join(owt_docs * 10) # 放大样本以获得更准确的测速
    raw_bytes = len(benchmark_text.encode("utf-8"))
    
    start_time = time.perf_counter()
    _ = owt_tokenizer.encode(benchmark_text)
    elapsed_time = time.perf_counter() - start_time
    
    throughput_bps = raw_bytes / elapsed_time
    throughput_mbps = throughput_bps / (1024 * 1024)
    
    pile_size_bytes = 825 * 1024 * 1024 * 1024
    estimated_seconds = pile_size_bytes / throughput_bps
    estimated_hours = estimated_seconds / 3600
    
    print(f"当前吞吐量: {throughput_mbps:.2f} MB/s")
    print(f"处理 825GB Pile 数据集预计耗时: {estimated_hours:.2f} 小时")

    # ---------------------------------------------------------
    # 实验 (d): 数据集序列化与 uint16 验证
    # ---------------------------------------------------------
    print("\n=== 实验 (d): 序列化数据集 ===")
    # 这里仅以 10 篇文档为例演示生成 npy 文件。正式实验时可以传入完整的迭代器。
    encoded_tokens = list(ts_tokenizer.encode_iterable(ts_docs))
    
    # 转换为 uint16 格式并保存
    uint16_array = np.array(encoded_tokens, dtype=np.uint16)
    np.save("tinystories_train_encoded.npy", uint16_array)
    
    print(f"验证 NumPy 数组数据类型: {uint16_array.dtype}")
    print("最大 Token ID:", np.max(uint16_array) if len(uint16_array) > 0 else 0)


if __name__ == "__main__":
    main()