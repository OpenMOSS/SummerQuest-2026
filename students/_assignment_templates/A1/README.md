# A1 公开提交：<姓名>

> 本文件和同目录代码公开可见。只提交允许公开且已经脱敏的内容；组织内材料放在下方
> 登记的飞书补充文档中，密钥和访问凭据不进入任何提交材料。

> 报告分节要求见 [`assignments/A1/README.md` 第 9 节](../../../assignments/A1/README.md#9-readme-报告内容要求)；
> 日志格式见[第 8 节](../../../assignments/A1/README.md#8-实验日志格式-logs)；
> 评分标准见[第 10 节](../../../assignments/A1/README.md#10-评分标准100-分)。把下面各占位处填好即可。

## 基本信息

- 作业题面版本：26.0.3
- 完成范围：<填写>
- 未完成项：<填写；没有则写“无”>
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（必须与 `SummerQuest-2026` 同级）

## Markdown 报告

> 以下小节标题请照抄，批改会按标题定位。曲线图放 `assets/` 并在正文引用。

### 书面题

- `unicode1`、`unicode2`：<简答>
- `adamw_accounting`：<AdamW 显存 / 最大 batch size / 单步 FLOPs / 训练时间核算>

数值答案同时填入下面的 `answers` 块（供自动核对）：

```json answers
{
  "unicode1_chr0": "<填写>",
  "unicode2_utf8_reason": "<填写>",
  "adamw_peak_memory_gpt2xl_bytes": null,
  "adamw_max_batch_size_80gb": null,
  "adamw_step_flops": "<填写>",
  "gpt2xl_train_hours_h100": null
}
```

### Tokenizer 实验

<最长 token、compression ratio、throughput，TinyStories 与 OWT 对比，交叉编码结论。>

### TinyStories 训练

<最终 val loss；贴 step 轴与 wall-clock 轴两张 loss 曲线。>

### 学习率扫

<多条学习率曲线，说明哪个 run 发散。>

### batch size 实验

<不同 batch size 的曲线与结论。>

### 消融：删除 RMSNorm

<曲线 + 分析。>

### 消融：Post-Norm

<曲线 + 分析。>

### 消融：NoPE

<曲线 + 分析。>

### 消融：SiLU

<曲线 + 分析。>

### OWT 实验

<曲线 + 与 TinyStories 的差异分析。>

### 文本生成

<至少 256 token 的样本 + 流畅度评价 + 至少两个影响因素。>

## 复现说明

- 环境与依赖：<填写公开、脱敏的信息>
- 数据准备：<填写公开数据的准备方法，不写内部路径>
- Tokenizer、训练与生成命令：<填写>
- 同步命令：`python3 scripts/sync_a1_submission.py --name '<姓名>'`
- 配置文件：<填写 submission/configs 下的相对路径；没有则写“无”>

## 代码与脚本

- 真实实现：`submission/cs336_basics/`
- 测试 adapter：`submission/tests/adapters.py`
- 训练、数据编码与生成脚本：`submission/scripts/`
- 实现说明：<填写>

真实实现先在兄弟目录 `../assignment1-basics` 中完成并通过官方测试，再使用同步命令复制
到本目录。不要手工复制公共 tests、fixtures、数据、模型权重、虚拟环境或依赖锁。

## 实验日志

- 日志目录：`logs/`
- 文件与格式：见 [`assignments/A1/README.md` 第 8 节](../../../assignments/A1/README.md#8-实验日志格式-logs)（固定文件名 + JSONL 字段 + `summary.json`）
- 与报告中实验的对应说明：<填写>

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有
审核必要的最小差量材料。
