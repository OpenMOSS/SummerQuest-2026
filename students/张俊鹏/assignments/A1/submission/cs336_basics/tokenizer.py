import json
import regex
from typing import Iterable, Iterator

class Tokenizer:
    
    GPT2_PATTERN = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    def __init__(self, vocab: dict[int, bytes], merges: list[tuple[bytes, bytes]], special_tokens: list[str] | None = None):
        """
        组装分词器引擎
        """
        self.vocab = vocab
        self.merges = merges
        self.special_tokens = special_tokens if special_tokens is not None else []
        self.special_tokens_set = set(self.special_tokens)
        
        # “反向词表”，用于编码时快速把 bytes 查成 ID
        self.inverse_vocab = {v: k for k, v in self.vocab.items()}
        
        # 给每个 merge 规则发一个“排位号（Rank）”
        self.merge_ranks = {pair: rank for rank, pair in enumerate(self.merges)}
        
        # 编译预处理的正则切刀
        self.compiled_pattern = regex.compile(self.GPT2_PATTERN)
        
        # 必须使用最长匹配优先

        if self.special_tokens:
            sorted_specials = sorted(self.special_tokens, key=len, reverse=True)
            escaped_specials = [regex.escape(tok) for tok in sorted_specials]
            self.split_pattern = regex.compile(f"({'|'.join(escaped_specials)})")
        else:
            self.split_pattern = None

    @classmethod
    def from_files(cls, vocab_filepath: str, merges_filepath: str, special_tokens: list[str] | None = None):
        """
        读取文件
        """
        # 假设 vocab 是保存为 json 的格式（键为字符串形式的数字，值为 iso-8859-1 或 base64，这里提供通用逻辑）
        # 实际使用时，请根据助教要求的持久化格式微调这里的加载代码
        with open(vocab_filepath, 'r', encoding="utf-8") as f:
            raw_vocab = json.load(f)
            # JSON 的 key 只能是字符串，需要转回 int；value 假设你之前存成了 ISO-8859-1 的字符串来避开 bytes 序列化问题
            vocab = {int(k): v.encode("iso-8859-1") for k, v in raw_vocab.items()}
            
        with open(merges_filepath, 'r', encoding="utf-8") as f:
            merges = []
            for line in f:
                # 假设 merges 是一行存一对，中间用空格隔开
                parts = line.strip('\n').split(' ')
                if len(parts) == 2:
                    merges.append((parts[0].encode("iso-8859-1"), parts[1].encode("iso-8859-1")))
                    
        return cls(vocab, merges, special_tokens)

    def encode(self, text: str) -> list[int]:
        """
        核心推断过程：将文本切碎，并按规矩合并成 Token ID 序列
        """
        ids = []
        
        # 1. 物理隔离：先切分 Special Tokens
        if self.split_pattern:
            pieces = self.split_pattern.split(text)
        else:
            pieces = [text]

        for piece in pieces:
            if not piece:
                continue
                
            # 如果是特殊标记，直接查出它的 ID 并塞进结果，绝不切碎！
            if piece in self.special_tokens_set:
                special_bytes = piece.encode("utf-8")
                ids.append(self.inverse_vocab[special_bytes])
                continue

            # 2. 正常文本：使用 GPT-2 正则切成小块 (chunks)
            chunks = self.compiled_pattern.findall(piece)
            for chunk in chunks:
                chunk_bytes = chunk.encode("utf-8")
                # 砸成最基础的单字节积木
                shattered = [bytes([b]) for b in chunk_bytes]

                # 3. 按照排位号 (Rank) 进行极速合并
                while len(shattered) >= 2:
                    # 找出当前 chunk 里所有的相邻积木对
                    pairs = [(shattered[i], shattered[i+1]) for i in range(len(shattered) - 1)]
                    
                    # 在这堆积木对里，找出在 merges 规则字典中 rank 最小（优先级最高）的那个对子
                    # 如果某个对子根本不存在于 merges 字典里，就给它一个无穷大 (float('inf')) 的排位
                    best_pair = min(pairs, key=lambda p: self.merge_ranks.get(p, float('inf')))
                    
                    # 如果最厉害的对子排位都是无穷大，说明没有任何积木可以合并了，直接结束当前 chunk 的合并
                    if self.merge_ranks.get(best_pair) is None:
                        break

                    # 真正执行合并：遍历碎积木，遇到冠军对子就粘起来
                    new_shattered = []
                    i = 0
                    while i < len(shattered):
                        if i < len(shattered) - 1 and shattered[i] == best_pair[0] and shattered[i+1] == best_pair[1]:
                            new_shattered.append(best_pair[0] + best_pair[1])
                            i += 2
                        else:
                            new_shattered.append(shattered[i])
                            i += 1
                    shattered = new_shattered

                # 将合并到极限的积木块，全部翻译成整数 ID 存起来
                for token_bytes in shattered:
                    ids.append(self.inverse_vocab[token_bytes])

        return ids

    def encode_iterable(self, iterable: Iterable[str]) -> Iterator[int]:
        """
        支持巨型文件的“懒加载”编码（极度节省内存）
        """
        for string in iterable:
            for token_id in self.encode(string):
                yield token_id

    def decode(self, ids: list[int]) -> str:
        """
        将一堆机器读的整数 ID，翻译回人类读的文本
        """
        raw_bytes = b"".join([self.vocab[idx] for idx in ids])

        return raw_bytes.decode("utf-8", errors="replace")