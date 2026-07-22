from train_bpe import train_bpe
import json
import os

def save_tokenizer(vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], vocab_path: str, merges_path: str):
    serializable_vocab = {str(k): v.decode("iso-8859-1") for k, v in vocab.items()}

    with open(vocab_path, "w", encoding="utf-8") as f:
        json.dump(serializable_vocab, f, ensure_ascii=False, indent=2)
    print(f"词表已成功保存至: {vocab_path}")

    with open(merges_path, "w", encoding="utf-8") as f:
        for p0, p1 in merges:
            line = f"{p0.decode('iso-8859-1')} {p1.decode('iso-8859-1')}\n"
            f.write(line)
    print(f"合并规则已成功保存至: {merges_path}")


if __name__ == "__main__":

    special_tokens = ["<|endoftext|>"]

    print("开始训练 BPE 分词器...")
    trained_vocab, trained_merges = train_bpe(
        input_path="data/owt_train.txt", 
        vocab_size=32000, 
        special_tokens=special_tokens
    )

    # 保存到磁盘
    save_tokenizer(
        vocab=trained_vocab, 
        merges=trained_merges, 
        vocab_path="vocab_owt.json", 
        merges_path="merges_owt.txt"
    )