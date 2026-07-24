# A1 补充审阅材料：许志鹏

> 用途：这是一份给飞书补充文档使用的本地草稿。
> 你可以直接把正文复制到飞书，再按文中给出的本地路径去找图片并粘贴。
> 本文件只做“审阅辅助材料”整理，不替代公开提交的 README。

---

## 1. 文档用途与对应公开材料

本补充文档对应以下公开提交内容：

- `students/许志鹏/assignments/A1/README.md`
- `students/许志鹏/assignments/A1/submission/`
- `students/许志鹏/assignments/A1/logs/`
- `students/许志鹏/assignments/A1/assets/`

GitHub README 负责给出最终公开结果与分析；本补充文档负责放置：

1. 实验运行截图
2. 训练曲线索引
3. 每组实验的简短备注
4. 失败 / 中断 run 的说明
5. README 中不便完整展开但仍需要审阅的过程记录

---

## 2. 训练环境与运行背景

### 2.1 开发与测试环境

- 代码实现仓库：`../assignment1-basics`
- 公开提交仓库：`SummerQuest-2026`
- Python 版本：3.12
- 主要依赖：`torch`、`numpy`、`pytest`、`pytest-timeout`、`psutil`、`regex`、`tiktoken`、`einops`、`jaxtyping`

### 2.2 训练环境

- 训练设备：H100 80GB
- 正式训练主要在远程 GPU 机器完成
- OWT tokenizer 和编码阶段大量使用 CPU
- OWT tokenizer 最终采用 Rust fast BPE 后端，以缩短数据准备时间

### 2.3 工程约束

本次 A1 的主要工程约束有两类：

1. OWT tokenizer 训练在 Python 实现下耗时过长
2. 大 batch TinyStories 实验容易逼近 H100 80GB 显存上限

因此补充了：

- Rust fast BPE 后端
- batch size 显存边界探索
- 失败 / 中断 run 的原始日志保留

---

## 3. 建议插入的图片与本地路径

> 说明：下面按实验组逐张列出建议插图。你不一定要全部贴满，但这份清单已经把“可用图片”和“推荐优先级”都整理好了。
> 图片主目录：
>
> `D:\文档\DOWNLOADS\png_images`

### 3.1 推荐优先插入的 12 张图

这 12 张最值得贴，基本已经能构成一份很完整的补充审阅文档。

#### 图 1：TinyStories baseline（保守 LR 起点）

路径：

`D:\文档\DOWNLOADS\png_images\20260711-173705-tinystories_baseline-vs10000-ctx256-dm512-l4-h16-bs128-lr0.0003\training_curves.png`

建议图注：

> 图 1：TinyStories 在较保守学习率 `3e-4` 下的训练曲线。该 run 已能稳定训练并达到基础目标，但明显不是最佳点。

#### 图 2：TinyStories 当前最佳 LR 点

路径：

`D:\文档\DOWNLOADS\png_images\20260712-120140-tinystories_lr_1e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.01\training_curves.png`

建议图注：

> 图 2：TinyStories `max_lr=1e-2` 的训练曲线。该点是本轮学习率 sweep 中的最佳结果。

#### 图 3：TinyStories 高 LR 不稳定示例 1

路径：

`D:\文档\DOWNLOADS\png_images\20260711-193644-tinystories_lr_3e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.03\training_curves.png`

建议图注：

> 图 3：TinyStories `max_lr=3e-2` 的训练曲线。高学习率已明显推高训练和验证损失，进入不稳定区。

#### 图 4：TinyStories 高 LR 不稳定示例 2

路径：

`D:\文档\DOWNLOADS\png_images\20260711-201111-tinystories_lr_5e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.05\training_curves.png`

建议图注：

> 图 4：TinyStories `max_lr=5e-2` 的训练曲线。该 run 作为 README 中 divergent 示例，用于说明高 LR 会显著破坏优化稳定性。

#### 图 5：TinyStories batch sweep 最优点

路径：

`D:\文档\DOWNLOADS\png_images\20260712-173635-tinystories_bs256_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01\training_curves.png`

建议图注：

> 图 5：TinyStories `bs256` 的训练曲线。该点是 batch size sweep 中的当前最佳设置。

#### 图 6：TinyStories 大 batch 但未 OOM

路径：

`D:\文档\DOWNLOADS\png_images\20260712-190929-tinystories_bs768_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs768-lr0.01\training_curves.png`

建议图注：

> 图 6：TinyStories `bs768` 的训练曲线。该设置已接近 H100 80GB 的实际可用 batch 上限，但泛化效果开始退化。

#### 图 7：TinyStories 大 batch 失败示例

路径：

`D:\文档\DOWNLOADS\png_images\20260712-190945-tinystories_bs896_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs896-lr0.01\training_curves.png`

建议图注：

> 图 7：TinyStories `bs896` 的训练曲线。该 run 在第一次 validation 后继续训练时触发 OOM，用于说明显存边界。

#### 图 8：No-RMSNorm 消融（高 LR）

路径：

`D:\文档\DOWNLOADS\png_images\20260713-063800-tinystories_ablate_no_rmsnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normnone\training_curves.png`

建议图注：

> 图 8：删除 RMSNorm、`max_lr=1e-2` 时的训练曲线。该 run 最终出现 NaN，说明 RMSNorm 对当前设置下的稳定训练非常关键。

#### 图 9：Post-Norm 消融（高 LR）

路径：

`D:\文档\DOWNLOADS\png_images\20260713-063806-tinystories_ablate_postnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normpospost\training_curves.png`

建议图注：

> 图 9：Post-Norm、`max_lr=1e-2` 时的训练曲线。训练在中后段明显失稳，best validation loss 远差于 baseline。

#### 图 10：Post-Norm 消融（低 LR）

路径：

`D:\文档\DOWNLOADS\png_images\20260713-081621-tinystories_ablate_postnorm_lr3e-3_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.003-normpospost\training_curves.png`

建议图注：

> 图 10：Post-Norm、`max_lr=3e-3` 时的训练曲线。降低学习率后训练恢复稳定，但效果仍不如 Pre-Norm baseline。

#### 图 11：OWT 保守 LR

路径：

`D:\文档\DOWNLOADS\png_images\20260713-142525-owt_bs256_lr3e-4-vs32000-ctx256-dm512-l4-h16-bs256-lr0.0003\training_curves.png`

建议图注：

> 图 11：OWT 在较保守学习率 `3e-4` 下的训练曲线，作为开放域训练的保守起点参考。

#### 图 12：OWT 当前最佳 run

路径：

`D:\文档\DOWNLOADS\png_images\20260713-174641-owt_bs256_lr3e-3-vs32000-ctx256-dm512-l4-h16-bs256-lr0.003\training_curves.png`

建议图注：

> 图 12：OWT 当前最佳 run 曲线。相较更保守 LR 取得更低 validation loss，但生成仍有 repetition loop。

---

### 3.2 TinyStories 学习率 sweep 全量图片索引

如果你想把 LR sweep 做得更完整，可以把下面这些图依次贴进去：

- `3e-4`
  - `D:\文档\DOWNLOADS\png_images\20260711-173705-tinystories_baseline-vs10000-ctx256-dm512-l4-h16-bs128-lr0.0003\training_curves.png`
- `1e-3`
  - `D:\文档\DOWNLOADS\png_images\20260711-174347-tinystories_lr_1e-3-vs10000-ctx256-dm512-l4-h16-bs128-lr0.001\training_curves.png`
- `2e-3`
  - `D:\文档\DOWNLOADS\png_images\20260711-181407-tinystories_lr_2e-3-vs10000-ctx256-dm512-l4-h16-bs128-lr0.002\training_curves.png`
- `3e-3`
  - `D:\文档\DOWNLOADS\png_images\20260711-181411-tinystories_lr_3e-3-vs10000-ctx256-dm512-l4-h16-bs128-lr0.003\training_curves.png`
- `5e-3`
  - `D:\文档\DOWNLOADS\png_images\20260711-183733-tinystories_lr_5e-3-vs10000-ctx256-dm512-l4-h16-bs128-lr0.005\training_curves.png`
- `8e-3`
  - `D:\文档\DOWNLOADS\png_images\20260711-183745-tinystories_lr_8e-3-vs10000-ctx256-dm512-l4-h16-bs128-lr0.008\training_curves.png`
- `1e-2`
  - `D:\文档\DOWNLOADS\png_images\20260712-120140-tinystories_lr_1e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.01\training_curves.png`
- `1.2e-2`
  - `D:\文档\DOWNLOADS\png_images\20260711-201212-tinystories_lr_1.2e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.012\training_curves.png`
- `1.4e-2`
  - `D:\文档\DOWNLOADS\png_images\20260712-120304-tinystories_lr_1.4e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.014\training_curves.png`
- `1.5e-2`
  - `D:\文档\DOWNLOADS\png_images\20260711-193633-tinystories_lr_1.5e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.015\training_curves.png`
- `3e-2`
  - `D:\文档\DOWNLOADS\png_images\20260711-193644-tinystories_lr_3e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.03\training_curves.png`
- `5e-2`
  - `D:\文档\DOWNLOADS\png_images\20260711-201111-tinystories_lr_5e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.05\training_curves.png`

建议至少保留其中 5 张：

- `3e-4`
- `1e-3`
- `1e-2`
- `3e-2`
- `5e-2`

---

### 3.3 TinyStories batch size sweep 全量图片索引

- `bs1`
  - `D:\文档\DOWNLOADS\png_images\20260712-175300-tinystories_bs1_short_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs1-lr0.01\training_curves.png`
- `bs32`
  - `D:\文档\DOWNLOADS\png_images\20260712-175854-tinystories_bs32_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs32-lr0.01\training_curves.png`
- `bs64`
  - `D:\文档\DOWNLOADS\png_images\20260712-173601-tinystories_bs64_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs64-lr0.01\training_curves.png`
- `bs128`
  - `D:\文档\DOWNLOADS\png_images\20260712-120140-tinystories_lr_1e-2-vs10000-ctx256-dm512-l4-h16-bs128-lr0.01\training_curves.png`
- `bs256`
  - `D:\文档\DOWNLOADS\png_images\20260712-173635-tinystories_bs256_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01\training_curves.png`
- `bs512`
  - `D:\文档\DOWNLOADS\png_images\20260712-175159-tinystories_bs512_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs512-lr0.01\training_curves.png`
- `bs768`
  - `D:\文档\DOWNLOADS\png_images\20260712-190929-tinystories_bs768_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs768-lr0.01\training_curves.png`
- `bs896`
  - `D:\文档\DOWNLOADS\png_images\20260712-190945-tinystories_bs896_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs896-lr0.01\training_curves.png`

说明：

- `bs1024` 没有对应曲线图，因为启动即 OOM，未形成有效训练曲线

建议至少保留其中 6 张：

- `bs32`
- `bs64`
- `bs128`
- `bs256`
- `bs768`
- `bs896`

---

### 3.4 TinyStories 架构消融全量图片索引

- `no_rmsnorm @ 1e-2`
  - `D:\文档\DOWNLOADS\png_images\20260713-063800-tinystories_ablate_no_rmsnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normnone\training_curves.png`
- `postnorm @ 1e-2`
  - `D:\文档\DOWNLOADS\png_images\20260713-063806-tinystories_ablate_postnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normpospost\training_curves.png`
- `no_rmsnorm @ 3e-3`
  - `D:\文档\DOWNLOADS\png_images\20260713-081558-tinystories_ablate_no_rmsnorm_lr3e-3_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.003-normnone\training_curves.png`
- `postnorm @ 3e-3`
  - `D:\文档\DOWNLOADS\png_images\20260713-081621-tinystories_ablate_postnorm_lr3e-3_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.003-normpospost\training_curves.png`
- `nope`
  - `D:\文档\DOWNLOADS\png_images\20260713-083337-tinystories_ablate_nope_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-posembnone\training_curves.png`
- `silu`
  - `D:\文档\DOWNLOADS\png_images\20260713-083345-tinystories_ablate_silu_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-dff2048-ffnsilu\training_curves.png`

建议至少保留其中 4 张：

- `no_rmsnorm @ 1e-2`
- `postnorm @ 1e-2`
- `postnorm @ 3e-3`
- `nope` 或 `silu`

---

### 3.5 OWT 训练全量图片索引

- `owt_bs256_lr3e-4`
  - `D:\文档\DOWNLOADS\png_images\20260713-142525-owt_bs256_lr3e-4-vs32000-ctx256-dm512-l4-h16-bs256-lr0.0003\training_curves.png`
- `owt_bs256_lr1e-3`
  - `D:\文档\DOWNLOADS\png_images\20260713-142600-owt_bs256_lr1e-3-vs32000-ctx256-dm512-l4-h16-bs256-lr0.001\training_curves.png`
- `owt_bs128_lr1e-3`
  - `D:\文档\DOWNLOADS\png_images\20260713-174630-owt_bs128_lr1e-3-vs32000-ctx256-dm512-l4-h16-bs128-lr0.001\training_curves.png`
- `owt_bs256_lr3e-3`
  - `D:\文档\DOWNLOADS\png_images\20260713-174641-owt_bs256_lr3e-3-vs32000-ctx256-dm512-l4-h16-bs256-lr0.003\training_curves.png`

建议 4 张都保留，因为 OWT 总共就这几组核心实验，完整贴出并不冗余。

---

### 3.6 终端 / 训练截图

如果你还有单独截的终端图、报错图、`nvidia-smi` 图，也很值得补进飞书。建议优先贴：

1. 官方测试通过截图
2. TinyStories best run 完成截图
3. `bs896` / `bs1024` OOM 截图
4. `no_rmsnorm` NaN 截图
5. OWT tokenizer / encode 完成截图
6. OWT 生成样本截图
7. `nvidia-smi` 显存占用截图

建议图注写法示例：

> 图 X：`bs896` 训练在第一次 validation 后继续训练时触发 OOM，说明该设置已逼近 H100 80GB 显存极限。

> 图 X：`no_rmsnorm` 消融实验训练日志截图。即使降低学习率，最终仍出现 NaN，说明 RMSNorm 对当前设置下的稳定训练非常关键。

> 图 X：OWT tokenizer 训练与编码完成截图。为减少大语料预处理时间，本次使用 Rust fast BPE 后端。

---

## 4. 曲线图与 README 结论对照表

### 4.1 `tinystories_baseline_curves.png`

对应 README 内容：

- TinyStories baseline 训练结果
- baseline loss 曲线趋势
- 后续 batch sweep / ablation 的参考基线

支持的结论：

- baseline 已稳定训练
- 可达到较好的 TinyStories validation loss
- 可生成结构完整的儿童故事样本

### 4.2 `tinystories_lr_5e-2_divergent_curves.png`

对应 README 内容：

- 学习率 sweep 中的不稳定 run 示例

支持的结论：

- 高 LR 会导致 train / val loss 被明显推高
- 当前稳定边界大致位于 `1.5e-2` 到 `3e-2` 之间

### 4.3 `tinystories_postnorm_curves.png`

对应 README 内容：

- 架构消融中 Post-Norm 的不稳定性分析

支持的结论：

- Post-Norm 在 `1e-2` 下不稳定
- 降到 `3e-3` 后可以恢复稳定，但仍差于 Pre-Norm baseline

补充说明：

- 这里没有单独再放 `no_rmsnorm` 曲线，是因为其主要信息是最终 NaN；该现象已经在原始日志和表格中清楚保留。

### 4.4 `owt_best_curves.png`

对应 README 内容：

- OWT 最佳 run 的训练趋势
- OWT 与 TinyStories 的差异分析

支持的结论：

- `owt_bs256_lr3e-3` 是当前最佳 OWT run
- 模型已学到开放域英文表面风格
- 当前阶段生成仍有 repetition loop

---

## 5. 每组实验的简短备注

### 5.1 TinyStories 学习率 sweep

- `3e-4`：稳定，可达到基础目标，但不是最优
- `1e-3`：明显优于 `3e-4`
- `2e-3` / `3e-3`：继续改善，保持稳定
- `5e-3` / `8e-3`：接近平台区
- `1e-2`：当前 TinyStories LR sweep 最优点
- `1.2e-2` / `1.4e-2` / `1.5e-2`：开始退化
- `3e-2` / `5e-2`：进入明显不稳定区

### 5.2 TinyStories batch size sweep

- `bs32`：训练慢，效果较差
- `bs64`：比 `bs32` 明显改善
- `bs128`：标准参考点
- `bs256`：当前 batch sweep 最优点
- `bs512`：可稳定训练，但略差于 `bs256`
- `bs768`：显存更高，但验证损失退化
- `bs896`：第一次 validation 后 OOM
- `bs1024`：启动即 OOM

### 5.3 TinyStories 架构消融

- baseline：Pre-Norm + RMSNorm + RoPE + SwiGLU
- `no_rmsnorm @ 1e-2`：最终 NaN
- `postnorm @ 1e-2`：前期下降，后期失稳
- `no_rmsnorm @ 3e-3`：降 LR 后仍 NaN
- `postnorm @ 3e-3`：恢复稳定，但仍弱于 baseline
- `nope`：稳定但效果变差
- `silu`：稳定且接近 baseline，但略差

### 5.4 OWT 训练

- `owt_bs256_lr3e-4`：保守起点
- `owt_bs256_lr1e-3`：明显优于 `3e-4`
- `owt_bs128_lr1e-3`：same-iteration 参考点
- `owt_bs256_lr3e-3`：当前 OWT 最佳点

---

## 6. 失败 / 中断 run 说明

### 6.1 `tinystories_bs896_lr1e-2`

- 不是启动即失败
- 第一次 validation 后继续训练时 OOM
- 说明该设置已逼近 H100 80GB 显存极限

对应原始日志：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs\raw_20260712-190945-tinystories_bs896_lr1e-2-vs10000-ctx256-dm512-l4-h16-bs896-lr0.01.jsonl`

### 6.2 `tinystories_bs1024_lr1e-2`

- 启动即 OOM
- 未进入有效训练阶段

### 6.3 `tinystories_ablate_no_rmsnorm_bs256`

- 最终 NaN
- 判定为训练不稳定，而不是一次性偶发波动

对应原始日志：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs\raw_20260713-063800-tinystories_ablate_no_rmsnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normnone.jsonl`

### 6.4 `tinystories_ablate_no_rmsnorm_lr3e-3_bs256`

- 即使显著降低 LR，仍最终 NaN
- 说明 RMSNorm 对稳定训练是关键组件

对应原始日志：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs\raw_20260713-081558-tinystories_ablate_no_rmsnorm_lr3e-3_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.003-normnone.jsonl`

### 6.5 `tinystories_ablate_postnorm_bs256`

- `1e-2` 下约 step 1500 后出现明显失稳

对应原始日志：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs\raw_20260713-063806-tinystories_ablate_postnorm_bs256-vs10000-ctx256-dm512-l4-h16-bs256-lr0.01-normpospost.jsonl`

---

## 7. README 中未完全展开的过程记录

### 7.1 为什么 OWT tokenizer 改成 Rust fast BPE

Python 版 BPE 在 OWT 上耗时过长，不适合当前资源环境，因此补充了：

- `submission/scripts/fast_bpe/src/main.rs`
- `submission/scripts/train_tokenizer_fast.py`

目标是：

- 不改变核心算法逻辑
- 保持与 Python 版 merge 顺序一致
- 显著缩短大语料 tokenizer 训练时间

### 7.2 为什么没有额外放 second-best OWT sample

当前最佳 OWT sample 已足够支撑以下结论：

- 模型学到开放域英文表面风格
- 生成仍容易 repetition loop

因此 second-best sample 的信息增量有限，没有额外放入 README。

### 7.3 关于 MFU 的记录

训练时观察到脚本打印的 MFU 偏低。结合当前实现，这是合理现象，原因包括：

- from-scratch Python / PyTorch 训练脚本调度开销
- validation 和 checkpoint 开销
- 数据加载与 host/device 协调开销
- 当前模型规模较小，难以吃满 H100 峰值算力

### 7.4 tokenizer longest token 的观察

TinyStories longest token：

- ` accomplishment`

OWT longest token：

- `ÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂÃÂ`

这说明：

- TinyStories 更干净、更规则、更易压缩
- OWT 更开放、更杂，也更容易包含网页编码噪声

---

## 8. 原始日志与图表路径索引

### 8.1 summary 日志

目录：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs`

主要文件：

- `tinystories_lr_sweep_stage.jsonl`
- `tinystories_batch_sweep_stage.jsonl`
- `tinystories_ablation_stage.jsonl`
- `owt_tokenizer_stage.jsonl`
- `owt_training_stage.jsonl`

### 8.2 raw 逐步日志

目录：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\logs`

命名规则：

- `raw_*.jsonl`

说明：

- 保留逐点 `step / loss / lr / elapsed_sec / tokens / val` 信息
- 用于复核 unstable / failed run 的过程性现象

### 8.3 图表文件

目录：

`D:\1\desktop\SummerQuest-2026\students\许志鹏\assignments\A1\assets`

文件：

- `tinystories_baseline_curves.png`
- `tinystories_lr_5e-2_divergent_curves.png`
- `tinystories_postnorm_curves.png`
- `owt_best_curves.png`

---

## 9. 建议助教优先审阅的材料

如果只快速审阅，建议优先看：

1. `README.md`
2. `assets/tinystories_baseline_curves.png`
3. `assets/tinystories_lr_5e-2_divergent_curves.png`
4. `assets/tinystories_postnorm_curves.png`
5. `assets/owt_best_curves.png`
6. `logs/tinystories_lr_sweep_stage.jsonl`
7. `logs/tinystories_batch_sweep_stage.jsonl`
8. `logs/tinystories_ablation_stage.jsonl`
9. `logs/owt_training_stage.jsonl`
10. 若需复核细节，再查看对应 `raw_*.jsonl`

---

## 10. 最终说明

本飞书补充文档仅作为公开 README 的审阅辅助材料，不单独构成最终结论。

最终评审应综合以下内容：

1. 公开 README
2. `submission/` 中的真实实现
3. `logs/` 中的 summary 与 raw 日志
4. `assets/` 中的曲线图
5. 本文档中的过程性补充说明
