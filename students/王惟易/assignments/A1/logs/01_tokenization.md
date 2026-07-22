# A1 学习记录 01：Tokenization

日期：2026-07-14
状态：核心代码里程碑完成

## 1. 里程碑结论

本阶段完成了 byte-level BPE 的训练、序列化、实验 runner，以及支持 special token 和流式输入的 Tokenizer。

当前已经满足的核心合同：

- `train_bpe` 能从 UTF-8 文本训练 byte-level BPE，正确处理 pre-token 边界、special-token 硬边界、频率并列时的字典序规则。
- 训练器从朴素全量重算，演进为倒排索引增量更新，再用惰性删除 heap 消除每轮全表寻找最大 pair 的瓶颈。
- `Tokenizer.encode` 与 GPT-2/tiktoken 的参考行为一致。
- `Tokenizer.decode` 能正确重组跨 token 的 UTF-8 字节，并用 U+FFFD 替换非法字节序列。
- `Tokenizer.encode_iterable` 通过 1 MB 附加内存限制测试。
- BPE 和 Tokenizer 的公开测试全部达到预期结果。

尚未把“完整 TinyStories / OpenWebText 实验”算作本里程碑完成条件。它们属于后续实验与报告收尾，不阻塞模型组件实现。

## 2. 学习路径

```text
Unicode code point 与 Python str
└── UTF-8 与 bytes
    └── GPT-2 regex pre-tokenization
        └── special token 硬边界
            └── 朴素 BPE 训练
                └── 选择性 merge
                    └── pair -> pre-token IDs 倒排索引
                        └── 增量维护 pair counts
                            └── lazy heap 选择最佳 pair
                                └── 无损序列化与实验 runner
                                    └── Tokenizer encode/decode
                                        └── special token 与 encode_iterable
```

这条路径的重点不是堆叠优化技巧，而是逐步回答三个查询：

1. `pair_counts[pair]`：这个 pair 当前出现多少次？
2. `pair_to_pretoken_ids[pair]`：修改这个 pair 会影响哪些 pre-token？
3. `pair_heap`：当前应该选择哪个 pair？

三个数据结构职责不同，不能互相替代。

## 3. 核心理论

### 3.1 Unicode、UTF-8 与 byte-level token

- Unicode code point 是字符的整数编号；UTF-8 是把 code point 编码成一个或多个 byte 的规则。
- ASCII 字符在 UTF-8 中通常只占一个 byte；非 ASCII 字符通常占多个 byte。
- 单独处理每个 byte 再 decode 是错误的，因为一个字符的 UTF-8 字节必须放在一起解释。
- `bytes` 的迭代元素是 `int`，若需要单字节 `bytes`，可使用切片或 `bytes([value])`。
- `bytes(n)` 表示创建长度为 `n`、内容全为零的 bytes；它与 `bytes([n])` 含义不同。

byte-level BPE 的优势是基础词表包含全部 256 种 byte，因此任何 Unicode 文本都不存在真正的 OOV 字符。

### 3.2 Pre-tokenization 的作用

GPT-2 regex 先把文本切成较粗的 pre-token。BPE 只允许在单个 pre-token 内 merge，不能跨越 pre-token 边界。

这样做有两个主要动机：

- 控制搜索空间，避免每次在整个原始字节流上统计相邻 pair。
- 保留空格、字母、数字、标点等结构，避免产生大量跨语言结构且难以复用的 token。

pre-tokenization 必须保持无损：按顺序拼回所有匹配结果应等于原文。否则模型训练和生成面对的并不是同一段文本。

### 3.3 Special token

`<|endoftext|>` 等 special token 同时具有两种身份：

- 它是词表中的一个完整 token。
- 它是训练和编码时的硬边界，不参与普通 pair 统计，也不能被拆成 `<|`、`endoftext`、`|>`。

构造 special-token regex 时需要：

- 使用 `re.escape`，避免 `|` 等字符被解释成正则语法。
- 将较长 special token 放在前面，使重叠 token 在同一位置优先匹配最长项。

### 3.4 BPE 训练规则

初始词表由 256 个单 byte token 和用户提供的 special tokens 构成。`vocab_size` 包含这两部分以及后续 merge 生成的 token。

每轮训练：

1. 找到全局频率最高的相邻 pair。
2. 频率相同时选择字典序更大的 pair。
3. 在所有相关 pre-token 中进行非重叠、从左到右的 merge。
4. 将合并结果加入 vocab，并记录 merge 创建顺序。

pair 计数按相邻位置计算，因此 `(a, a, a)` 含有两个 `(a, a)`。如果该 pre-token 出现两次，全局贡献就是 4。

### 3.5 倒排索引与增量更新

训练过程中保留：

- `pretokens[id]`：该 pre-token 当前的 BPE 分解。
- `frequencies[id]`：该 pre-token 在语料中的固定出现次数。
- `pair_counts[pair]`：该 pair 的加权全局出现次数。
- `pair_to_pretoken_ids[pair]`：包含该 pair 的 pre-token ID 集合。

关键不变量：

```text
pair_counts[p]
= sum(frequencies[i] * occurrences(p, pretokens[i]))

pair_to_pretoken_ids[p]
= {i | occurrences(p, pretokens[i]) > 0}
```

“倒排”的含义是把自然的 `pre-token -> pairs` 查询方向反过来，建立 `pair -> pre-token IDs`。选择一个 pair 后，就能直接找到需要重建的 pre-token，而不用扫描全部 pre-token。

一次 indexed merge 的更新顺序：

1. 复制受影响 ID 集合，避免边遍历边修改索引。
2. 减去旧 pre-token 的局部 pair 贡献，并移除旧索引关系。
3. merge 当前 pre-token。
4. 加入新局部 pair 贡献和新索引关系。
5. 删除计数为零的 pair 与空索引集合。

### 3.6 Heap 与惰性删除

倒排索引解决了“更新哪些 pre-token”，但 `max(pair_counts, ...)` 仍会在每轮 merge 扫描所有 pair。

最终实现增加了候选 heap：

- Python `heapq` 是最小堆，因此候选对象反转 `(count, pair)` 的比较含义，让频率更高、同频时字典序更大的项位于堆顶。
- pair 计数改变时，不在 heap 内查找和原地修改旧 entry，而是压入最新 entry。
- 弹出时以 `pair_counts` 为事实来源；entry 的缓存计数若与当前计数不同，就丢弃并继续。

heap 允许重复 entry。重复且仍有效的 entry 代表同一个正确候选，只增加少量空间和弹出成本；计数变化后，旧副本会自然成为 stale entry。

复杂度由每轮扫描所有 pair 的近似 `O(MP)`，变为初始化 heap 后对 changed pairs 做 `O(log H)` 的 push/pop，其中 `M` 是 merge 数、`P` 是 pair 种类数、`H` 是 heap 大小。

### 3.7 Tokenizer 编码与解码

训练得到的 merges 已经固定了规则优先级。编码时不再统计局部频率，而是：

1. 使用与训练相同的 regex 预分词。
2. 把 pre-token 编码成单 byte token。
3. 在当前相邻 pair 中选择 merge rank 最小者。
4. 重复 merge，直到没有适用规则。
5. 将最终 byte tokens 映射成 token IDs。

rank 小表示规则创建得更早，并不一定表示语义上更“简单”。后续 merge 往往依赖早期 merge 生成的 token，因此编码必须严格重放优先级。

解码时必须先拼接全部 token bytes，再统一执行 UTF-8 decode。一个合法字符的多个 byte 可能分散在不同 token 中；逐 token decode 会把本可组合的字节错误替换成 U+FFFD。

## 4. 实现结构

### `cs336_basics/bpe.py`

- 词表初始化、pre-token 统计和 pair 统计。
- 朴素版本作为学习参考保留。
- `merge_pretoken` 实现从左到右、非重叠 merge。
- indexed state 初始化与增量 merge。
- heap 候选排序、初始化和惰性弹出。
- 生产版 `train_bpe`。

### `cs336_basics/bpe_io.py`

- 使用 JSON 保存 vocab 与 merges。
- bytes 使用十六进制字符串表示，避免假设单个 token 是合法 UTF-8。
- roundtrip 能恢复 token ID、任意 bytes 和 merge 顺序。

### `cs336_basics/bpe_experiment.py`

- 命令行参数解析。
- 记录训练时间与 Linux peak RSS。
- 输出最终词表大小、merge 数、最长 overall token 和最长 learned token。
- 训练计时不包含序列化。

### `cs336_basics/tokenizer.py`

- vocab 与反向 `bytes -> id` 映射。
- `pair -> merge rank` 映射。
- 普通文本预分词和 BPE encode。
- special-token 最长优先分段。
- 拼接 bytes 后统一 decode。
- 面向文件行迭代器的惰性 `encode_iterable`。

### `tests/adapters.py`

- adapter 只负责把公开测试接到学生实现，不承载算法逻辑。

## 5. 性能演进

以下结果来自笔记本 CPU；BPE 和 Tokenizer 阶段不需要 GPU。

| 数据与配置 | 实现 | 时间 | Peak RSS | 结果 |
|---|---:|---:|---:|---|
| `corpus.en`, vocab 500 | naive | 1.406 s | 未记录 | 243 merges |
| `corpus.en`, vocab 500 | final heap | 0.159 s | 未记录 | 与 naive 完全一致，约 8.8x |
| TinyStories 5M sample, vocab 1,000 | indexed | 1.412 s | 48.34 MiB | 743 merges |
| TinyStories 5M sample, vocab 10,000 | indexed + 全表 `max` | 4.777 s | 48.90 MiB | 9,743 merges |
| TinyStories 5M sample, vocab 10,000 | final heap | 2.028 s | 49.55 MiB | 与旧 JSON 逐字节一致 |

5M / vocab 10,000 的最长 learned token 是 `b' congratulations'`，长度 16 bytes。前导空格来自 GPT-2 pre-token pattern 的可选空格，符合预期。

heap 优化前的 profile：

- `choose_best_pair` 累计约 7.489 s。
- key lambda 被调用 53,695,350 次，平均每轮 merge 扫描约 5,511 个 pair。
- `count_pretokens` 累计约 2.448 s。
- `apply_indexed_merge` 累计约 0.384 s。

heap 优化后的 profile：

- `count_pretokens` 累计约 2.511 s，成为新瓶颈。
- `apply_indexed_merge` 累计约 0.412 s。
- `pop_best_pair` 累计约 0.180 s。
- 58,285 次 heap pop 对应 9,743 次 merge，平均约 6 次 pop/merge，惰性删除成本可接受。

注意：cProfile 会放大 Python 函数调用开销，profile 的绝对时间不能直接与未插桩运行比较；调用次数和热点排序更有解释力。

## 6. 验证证据

### BPE trainer

```text
tests/test_train_bpe.py::test_train_bpe_speed PASSED
tests/test_train_bpe.py::test_train_bpe PASSED
tests/test_train_bpe.py::test_train_bpe_special_tokens PASSED

3 passed in 1.67s
```

额外验证：

- final heap 与 naive 在 `corpus.en`, vocab 500 上得到完全相同的 vocab 和 merge 序列。
- 5M / vocab 10,000 的 heap 产物与优化前 JSON 逐字节一致。
- 最终 vocab 10,000，merge 数为 `10000 - 256 - 1 = 9743`。
- learned tokens 中不存在 `<|` 或 `|>` special-token 碎片。

### Tokenizer

```text
24 passed, 1 xfailed in 8.25s
```

覆盖：

- 空输入、ASCII、Unicode 和 emoji roundtrip。
- 与 GPT-2/tiktoken ID 序列一致。
- special token、连续 special token、重叠 special token。
- 地址、德语和 TinyStories fixture。
- special token 前后的换行边界。
- `encode_iterable` 与完整编码一致。
- `encode_iterable` 通过 1 MB 附加内存限制。

唯一 xfail 是题目明确预期的 `Tokenizer.encode` 1 MB 内存测试：普通 `encode` 返回完整 ID 列表，本来就不要求在该限制下通过。

### 静态检查

```text
ruff: passed
ty: passed
```

检查范围包括 tokenizer、BPE 实现和 adapter 接线。

## 7. 典型错误与调试经验

### Python 数据模型

- `bytes` 迭代得到 `int`，不是 `bytes`。
- `bytes(i)` 与 `bytes([i])` 含义不同。
- 字典查询使用 `mapping[key]`，不能把字典写成函数调用。
- `if not mapping[key]` 同时混淆“key 不存在”和“值为 0”；ID 或 rank 可能合法地等于 0，应使用成员判断或 `is not None`。
- `heapq.heapify` 原地修改列表并返回 `None`。
- 空 set 是 `set()`，`{}` 是空 dict。

### 算法状态

- indexed merge 要同时维护 pre-token、pair count 和倒排索引；只改其中一个会破坏全局不变量。
- 一个 pair 在同一 pre-token 内可能重复出现；计数需要保留 multiplicity，索引只需要记录受影响 ID 一次。
- heap 中的缓存不是事实来源，`pair_counts` 才是。
- `_encode_pretoken` 每完成一次 merge 都必须重新清空并寻找 best pair，否则旧选择会残留并造成死循环。

### 测试与定位

- smoke test 自己也可能写错。应从 traceback 的最早异常位置判断是 fixture 构造失败，还是被测逻辑失败。
- `repr` 对观察 NUL、空格和 bytes 很重要；直接打印某些字符会产生误导。
- 优化前先 profile；修复热点后重新 profile，因为瓶颈会迁移。
- 性能优化必须用 naive oracle 或旧产物做差分验证，不能只看“运行更快”。

## 8. 已知边界与暂缓事项

这些事项尚未完成，但不阻塞模型组件：

- 尚未在完整 TinyStories train 集上训练 vocab 10,000 tokenizer。
- 尚未运行 OpenWebText vocab 32,000 实验。
- 尚未完成两个语料 tokenizer 的 compression ratio、cross-domain compression 和吞吐量实验。
- 尚未把训练集与 validation 集序列化成 `uint16` token ID 数组。
- 尚未实现 multiprocessing pre-tokenization。题面硬限制是完整 TinyStories 不超过 30 分钟、30 GB RAM；“低于 2 分钟”只是提示。只有完整实测触及硬限制时再补并行。
- `encode_iterable` 已满足文件行迭代器和公开内存测试；若未来要支持任意位置切断的字符串 chunk，需要增加安全尾部缓冲。
- `Tokenizer.from_files` 便利接口尚未补齐；当前已有 `bpe_io.load_bpe`，可在最终收尾时薄封装。

## 9. 下一里程碑：模型组件

接下来按依赖顺序学习和实现：

```text
Linear
└── Embedding
    └── RMSNorm
        └── SiLU / SwiGLU
            └── RoPE
                └── scaled dot-product attention
                    └── multi-head self-attention
                        └── Transformer block
                            └── Transformer LM
```

每个节点继续采用相同节奏：先明确张量形状和数学合同，再完成最小实现，然后通过 snapshot/参考测试，最后才进入下一节点。
