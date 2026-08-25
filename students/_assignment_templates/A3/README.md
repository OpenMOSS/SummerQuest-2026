# A3 公开提交：<姓名>

> 本文件和同目录分析代码、轻量结果、依赖声明与图片公开可见。只提交允许公开且已经脱敏的
> 内容；API 地址、密钥、内部路径、完整服务日志和隐藏评测信息不得进入 GitHub 或 Git 历史。

> 正式要求见 [`assignments/A3/README.md`](../../../../assignments/A3/README.md)，评分说明见
> [`assignments/A3/EVALUATION.md`](../../../../assignments/A3/EVALUATION.md)。

## 基本信息

- 作业题面版本：`26.2.0`
- 完成范围：<填写>
- 未完成项：<填写；没有则写“无”>
- 探索预算：reserved <填写> 秒；charged <填写> 秒；remaining <填写> 秒
- 分析版本：<填写 commit SHA 或其他稳定版本标识>

## 1. 实验设计与预算

### 假设与规模覆盖

<说明各组实验要回答的问题、控制变量、模型规模、训练 token、预算分配和外推区域。>

### 正式实验表

正式、脱敏的机器可读记录位于 `results/experiments.csv` 或
`results/experiments.jsonl`。在这里汇总 experiment ID、状态、配置、runtime、validation
loss、fit role 和排除理由；失败与 partial run 不得伪装成 completed observation。

<填写关键实验表或对机器可读记录的说明。>

## 2. Scaling-law 拟合

### 变量、函数形式与估计方法

<定义 target、features、单位、变换、函数形式、参数约束、初始化、objective、weighting 和
估计方法。>

### 拟合结果

<引用 `results/fit_summary.json`，解释参数、有效拟合点数量和主要诊断。>

![Scaling-law 拟合与外推](assets/fit-extrapolation.png)

## 3. 诊断与敏感性

<报告 residual、留出预测、bootstrap、posterior predictive、替代模型或 run inclusion
敏感性；说明模型选择与外推风险。>

![Residual 或敏感性诊断](assets/diagnostics.png)

## 4. 最终配置与预测

- 最终配置公开摘要或 hash：<填写；不得写内部路径或隐藏字段>
- `predicted_final_loss`：<填写>
- `predicted_final_loss_lower`：<填写>
- `predicted_final_loss_upper`：<填写>
- 80% 区间构造方法：<填写>
- API `GET /final_submission` 核对结果：<填写已核对的脱敏摘要>

机器可读版本位于 `results/final_prediction.json`，必须与截止时间前 API 中的最新有效记录
一致。

## 5. 复现说明

- 依赖声明：<填写 `requirements.txt` 或 `pyproject.toml`>
- 最小运行命令：<填写从 `results/` 重建拟合、预测和两张图片的命令>
- 主要入口：<填写 `analysis/` 中的 Python 文件>
- 输出与报告对应关系：<填写>

`analysis/` 不得依赖私有绝对路径、未提交 notebook state、内部数据库或无法获得的数据。

## 6. 限制与风险

<说明可能失效的假设、未覆盖的不确定性、训练失败风险，以及看到真实 final result 前最担心的
问题。>

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，不得开启互联网公开访问，只保存不能公开到 GitHub 但确有审核必要
的最小差量材料；API key、Token、Cookie、密码和隐藏评测内容仍不得写入飞书正文。

## 自检

- [ ] 本 PR 只包含我本人本次 A3 的文件。
- [ ] `analysis/` 至少包含一个可执行 Python 文件，并可从提交的 `results/` 重建主要结果。
- [ ] 已提交且只提交一个 `experiments.csv` 或 `experiments.jsonl`。
- [ ] `fit_summary.json` 和 `final_prediction.json` 满足正式题面的 schema。
- [ ] 至少两张报告图片位于 `assets/`，已经压缩并由 README 使用相对路径引用。
- [ ] 已提交轻量 `requirements.txt` 或 `pyproject.toml`，未提交锁文件和完整依赖环境。
- [ ] README、JSON 与冻结前 API final submission 一致。
- [ ] `results/` 与 `assets/` 合计不超过 2 MiB，README 不超过 1 MiB。
- [ ] 未提交凭据、内部 URL/路径、数据、权重、manifest、snapshot、backend 或完整日志。
- [ ] 飞书补充文档为组织内公开，且未开启互联网公开访问。
