# 任务一：Activation Checkpointing
## 1.1 gradient_checkpointing 的理论部分
在忽略计算成本的意义下，完全重算可以达到 O(1) 内存，但是不满足使用嵌套checkpoint的隐含要求

### 1. 检查点安排与是否嵌套
采用 递归二分 + 嵌套 checkpoint 。把整个 [0, N) 序列作为最外层 checkpoint，内部每个子段继续用 checkpoint 包裹，直到子段长度为 1 时直接计算一个 Transformer block。

- 嵌套方式 ：每个非叶子节点 [l, r) 先 checkpoint 左半段 [l, mid) ，得到中点边界激活 mid_out ；再以 mid_out 为输入 checkpoint 右半段 [mid, r) 。
- 不嵌套时 （如逐层 checkpoint）：每 block 保存一个输入，前向结束后 N 个输入同时驻留，峰值仍是 O(N)。
- 嵌套后 ：任一时刻只需保留递归路径上的 O(log N) 个中点边界激活，以及当前正在 backward 的子树内部极少量激活。
### 2. 渐近分析
指标 复杂度 说明 峰值 activation memory O(log N) 递归树深度为 log N，每层保留一个中点边界激活 总计算量 O(N log N) 每个 block 在从叶子到根的 log N 层祖先 backward 中各被重算一次

### 3. 伪代码骨架（≤ 20 行）
```
def forward_segment(l, r, x):
    if r - l == 1:                    # 叶子：单个 block，不嵌套 checkpoint
        return block[l](x)
    mid = (l + r) // 2
    # checkpoint 左半段，保存边界激活 mid_out
    mid_out = checkpoint(forward_segment, l, mid, x)
    # checkpoint 右半段，以 mid_out 为边界输入
    return checkpoint(forward_segment, mid, r, mid_out)

# 最外层：checkpoint 整个 [0, N)
output = checkpoint(forward_segment, 0, N, x)
```
### 4. 边界 activation、重计算区间与峰值位置
**保存的边界 activation**  
只有被 checkpoint(...) 包裹的子段 输入 会被保留：

- 最外层保存原始输入 x ；
- 每个内部父节点保存左半段的输出 mid_out ；
- 叶子 block 不保存内部激活，只返回输出。
**重计算区间**  
Backward 时，每当一个 checkpoint 节点被反向传播触及，就重算对应子段的 forward：

- 最外层 backward 重算整个 [0, N) ；
- 内部每个 checkpoint 节点被其上层 backward 触发时，重算自己的子段；
- 叶子 block 不触发重算，直接计算梯度。
**峰值出现位置** 
峰值内存出现在 某个中间子树被 backward 的时刻，例如对 [N/2, N) 做 backward 时：

- 路径上已保存 O(log N) 个祖先中点输入（ x , mid_out_1 , ...）；
- 当前子树内部只展开一个 2-block 叶子对（O(1) 激活）；
- 已处理完的右子树完全释放，未处理的左子树只保留根输入。
因此峰值 activation memory = O(log N) 个边界输入 + O(1) 当前激活 = O(log N) 。

## 1.2 固定实验
### 实验设置

- Model: Stanford medium（d_model=1024，num_layers=24）
- Batch size: 1，sequence length: 1024（标准矩阵）/ 2048（边界测试）
- BF16 autocast，FP32 参数，AdamW
- 3 warmup + 5 measurement step
- 23 GiB allocator guard

### 结果

原始文件位于"assignment2-systems/local_results/checkpointing.csv"

1024 标准矩阵：

| block_size | peak_allocated (MiB) | peak_reserved (MiB) | p50 step time (ms) |
|------------|---------------------:|--------------------:|-------------------:|
| 0 (无 ckpt) | 10133 | 10264 | 208 |
| 1 | **6886** | 7058 | 321 |
| 2 | 7026 | 7178 | 316 |
| 4 | 7305 | 7440 | 297 |
| 8 | 7862 | 7958 | 283 |

2048 边界测试：

| block_size | peak_allocated (MiB) | p50 step time (ms) | status |
|------------|---------------------:|-------------------:|--------|
| 0 | — | — | OOM |
| 1 | 8105 | 488 | success |

---

### 分析：最佳 block size 不由 checkpoint 数量单独决定

显存由两部分组成：
1. **保存的 boundary activation**：block_size 越小，保存的输入越多
2. **重算时的中间激活**：block_size 越大，每段越长，同时存在的激活越多

实验数据中，block_size=1 显存最低，不是因为它保存的输入少，而是因为**每段只有 1 层，重算激活最小**。block_size=8 虽然只保存 3 个边界输入，但每段 8 层的激活在 backward 时同时驻留，反而显存更高。

### 显存收益与重计算代价

- block_size=1：显存最低（−32%），但时间 +55%
- block_size=4：显存 −28%，时间 +43%
- block_size=8：显存 −22%，时间 +36%

2048 长度下，无 checkpoint 直接 OOM，block_size=1 成功，说明 checkpoint 对长序列训练是有必要的。

### 结论

最佳 block_size 需要在**显存收益**和**重计算开销**之间权衡，不是 checkpoint 数量越多越好。block_size 过小调度开销大，过大则激活驻留多。

# 任务二：PyTorch Attention 与 torch.compile
## 显式 PyTorch 基线

源文件在assignment2-systems/local_results/attention_baseline.csv

| seq_len | head_dim | forward (ms) | backward (ms) | forward-backward (ms) | peak_allocated (MiB) |
|---------|----------|-------------|--------------|----------------------|---------------------|
| 512 | 64 | 0.045 | 0.588 | 1.606 | 275 |
| 512 | 128 | 0.071 | 1.187 | 2.305 | 275 |
| 2048 | 64 | 0.105 | 1.257 | 2.297 | 310 |
| 2048 | 128 | 0.164 | 1.232 | 2.452 | 312 |
| 8192 | 64 | 1.860 | 2.468 | 4.253 | 855 |
| 8192 | 128 | 1.881 | 2.508 | 4.334 | 861 |

**关键观察**：
- 显存主要随 `seq_len²` 增长，head_dim 影响很小
- 8192 时显存跳到 ~860 MiB，符合 attention score 矩阵 `(8192, 8192)` 的大小
- backward 时间约为 forward 的 10 倍以上（因为反向要保存/重算 score 矩阵）
