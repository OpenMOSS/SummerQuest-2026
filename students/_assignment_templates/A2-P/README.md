# A2-P 公开提交：<姓名>

> 本文件和同目录代码、汇总、图片公开可见。只提交允许公开且已经脱敏的内容；大型 profiler
> 原始文件留在个人工作目录，组织内差量材料放入下方登记的飞书补充文档。密钥和访问凭据
> 不进入任何提交材料。

> 正式要求见
> [`assignments/A2-P/README.md`](../../../../assignments/A2-P/README.md)，评分说明见
> [`assignments/A2-P/EVALUATION.md`](../../../../assignments/A2-P/EVALUATION.md)。

## 基本信息

- 作业题面版本：`26.1.4-rc.3`
- 完成范围：<填写>
- 未完成项：<填写；没有则写“无”>
- 上游 starter commit：`ca8bc81a59b70516f7ebb2da4808daade877c736`
- 本地工作仓库：`../assignment2-systems`

## 环境与工具

| 项目 | 公开、脱敏的信息 |
| --- | --- |
| GPU | <填写型号，不写 UUID、主机名或内部资源编号> |
| Driver / CUDA | <填写> |
| PyTorch | <填写> |
| Compute profiler | <填写 Nsight Systems 或 torch.profiler 版本> |
| 其他限制 | <填写；没有则写“无”> |

## 1. End-to-End Benchmark

### 复现命令与计时方法

<填写命令、三种 mode、同步位置、warm-up 与 measurement 边界。>

### 结果

<引用 `results/benchmark.csv`，给出 raw timings、均值、标准差、CV 和 warm-up 对照。>

### 分析

<解释差异与不确定性。>

## 2. Compute Profiling

### 六个 `train_step` trace 与命令

<填写两个模型规模、三个 context、train_step、dtype、所选工具、阶段标记和命令。>

### Kernel、Calls 与时间线

<引用 `results/profile/` 和一张裁剪、压缩的关键 Profile 截图，解释主要 op/kernel、
Calls、CPU/CUDA 时间和阶段对应关系。>

### 工具边界

<若使用 nsys，说明 CUDA API 与 kernel 关联；若使用 torch.profiler，说明 schedule、
CPU/CUDA activities、Perfetto 阅读方式以及与 nsys 的证据边界差异。>

## 3. Mixed Precision

### 四种累加实验

<填写实际输出，并区分输入量化误差与累加器误差。>

### FP32 与 BF16 autocast

<引用 `results/mixed_precision.json`，确认 ToyModel 使用 BF16 autocast，填写 dtype、
时间、显存和趋势。>

## 4. Memory Profiling

### 配置、峰值与 fallback

<引用 `results/memory/peaks.csv`。若发生 OOM，写明原配置、失败阶段和实际 fallback。>

### Timeline、allocation 与 residual/gradient

<至少引用两张 `assets/` 图片，解释 active/allocated/reserved、最大 allocation 和
TransformerBlock 中 residual/gradient 的变化。>

## 5. 限制与复现

- 代码同步命令：`python3 scripts/sync_a2p_submission.py --name '<姓名>'`
- 轻量结果目录：`results/`
- 未提交的本地大型原始文件：<只写类型和本地保留策略，不写内部路径>
- 已知限制：<填写>
- 最小复现步骤：<填写>

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有审核必要
的最小差量材料；不要机械复制公开报告，也不要随意上传大型 trace、snapshot 或凭据。

## 自检

- [ ] 本 PR 只包含我本人本次 A2-P 的文件。
- [ ] `README.md` 是 Markdown 主报告，所有图片使用相对路径和有意义的 alt text。
- [ ] 每个关键数字都能回到命令、`results/` 或 metadata。
- [ ] 引用仓库外源码或资料时使用固定 commit 的 GitHub HTTPS 绝对 URL，未写入本机路径或 `file://` 链接。
- [ ] 已用 nsys 或 `torch.profiler` 完成六个 `train_step` trace，并提交轻量汇总。
- [ ] 已提交 1 张 Compute Profile 关键图和至少 2 张 Memory Timeline，均已裁剪、压缩并被报告引用。
- [ ] `results/` 与 `assets/` 公开附件合计不超过 2 MiB。
- [ ] 未提交 `.nsys-rep`、snapshot、完整 trace、权重、数据、压缩包或依赖环境。
- [ ] GitHub 内容不含内部主机名、IP、账号、路径、UUID、进程或未公开项目。
- [ ] GitHub 和飞书正文都不含 Secret、Token、Cookie、密码或私钥。
- [ ] 飞书补充文档为组织内公开，且未开启互联网公开访问。
