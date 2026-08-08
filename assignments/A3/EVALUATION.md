# A3 评估补充说明（批改助教）

> 本文件说明 A3 的评分、核验与异常处理方式，不改变
> [`README.md`](README.md) 中的任务、预算、提交和公开性要求。若本文与 README 或正式
> 课程通知冲突，以正式课程通知和 README 的较新版本为准。

## 1. 评估原则

- 评估链路是“正式 API run → 轻量实验表 → 分析代码 → 拟合与诊断 → 最终配置、点预测和
  区间 → 集中 final run”。任何关键结论都应能沿这条链路反向核验。
- **actual final validation loss**、**prediction quality** 和 **methodology** 是三个独立
  维度。最终配置 loss 低不自动证明拟合方法有效；方法写得完整也不能替代失败的最终 run。
- 方法评分主要检查两件事：已有实验是否构成有效证据，以及拟合/外推方法是否被这些证据
  支持。不按公式复杂度、图表数量或术语多少给分。
- 预测区间按 80% central prediction interval 解释。评分同时惩罚区间宽度和漏掉真实 loss
  的距离，不以“覆盖即可”为标准。
- 评分时使用截止时间冻结的 API final submission 和报告 commit。结果公布后的修改不回写
  原始评分证据。
- 平台、调度、存储或课程 backend 故障与学生配置失败分开处理。确认的系统故障不自动按
  最差学生结果计分，但必须由课程团队审计后决定重跑或裁定。

## 2. 总体权重

| 部分 | 权重 | 原始评估量 |
| --- | ---: | --- |
| 最终 run 的 actual validation loss | 60% | held-out final evaluation 上的实际 loss，越低越好 |
| 点预测与 80% prediction interval | 20% | `prediction_quality_penalty`，越低越好 |
| 实验、拟合与复现性 | 20% | 下文 methodology rubric |

actual-loss 与 prediction-quality 的原始数值、有效/无效状态和排名应一并保留在评分导出中。
如需把 lower-is-better 原始量转换为班级内的有界分数，转换规则必须在 final submission
冻结前由课程团队统一确定，并对所有有效提交使用同一规则；批改助教不得在看到结果后为个别
同学调整归一化方法。

## 3. Actual final validation loss（60%）

集中评测只使用冻结的最新有效 final submission。课程 backend 在一致的最终 runtime、训练
栈、数据工件和 held-out evaluation 上执行一次正式 run。

有效结果必须满足：

1. final submission 在截止时间前通过 schema 和区间校验；
2. 正式任务状态为 `completed`；
3. worker/backend 记录了有限的 `actual_final_validation_loss`；
4. manifest、提交配置和结果记录能够按 final run ID 对齐；
5. 没有确认的完整性、越权或学术诚信问题。

同一批次中 actual validation loss 越低，该部分表现越好。不得使用 exploratory evaluation
loss 代替 held-out final loss，也不得对失败任务外推一个“假想 final loss”。

学生配置导致的 timeout、OOM/resource exhaustion、numerical instability、非法 shape 或其他
训练失败，final-loss 部分按无有效结果并进入最低档处理；prediction 部分因为没有可信 actual
loss，也不能按正常公式自动得到有效分。确认的平台故障先标记为 `staff_review`，完成重跑或
裁定后再计分。

## 4. Prediction quality（20%）

令实际最终 validation loss 为 $L$，学生点预测为 $\hat{L}$，80% 中心预测区间为
$[L_-, L_+]$。有效预测必须全部为有限数，并满足：

$$
L_- \leq \hat{L} \leq L_+.
$$

### 4.1 点预测误差

$$
E_{\text{point}} = |\hat{L} - L|.
$$

### 4.2 区间分数

$$
S_{\text{interval}}
=
(L_+ - L_-)
+ 10\max(0, L_- - L)
+ 10\max(0, L - L_+).
$$

第一项惩罚区间宽度；第二、三项在真实 loss 落到区间外时惩罚 miss distance。系数 10 来自
80% 中心预测区间的标准 interval score：$2 / 0.2 = 10$。

### 4.3 综合 prediction penalty

$$
S_{\text{pred}}
=
0.5 E_{\text{point}}
+ 0.5 S_{\text{interval}}.
$$

`S_pred` 越低越好。评分导出至少记录：

- `predicted_final_loss`；
- `actual_final_validation_loss`；
- `prediction_absolute_error`；
- interval lower、upper 和 width；
- interval 是否覆盖实际 loss；
- interval miss distance；
- `interval_score`；
- `prediction_quality_penalty`。

助教不得在看到 $L$ 后替学生修正 point prediction、交换上下界、扩大区间或选择 README 中的
另一组数字。README、`final_prediction.json` 与 API 冻结记录不一致时，以 API 冻结记录为
正式预测，并在复现性评分中记录不一致。

## 5. Methodology（20%）

| 子项 | 分值 | 核验重点 |
| --- | ---: | --- |
| 正式证据与实验设计 | 7 | API provenance、规模覆盖、控制变量、预算分配、run 角色与失败处理 |
| 拟合与外推方法 | 7 | 变量定义、函数形式、约束、估计、诊断、替代模型与最终配置搜索 |
| 区间构造与风险分析 | 4 | 80% 区间方法、calibration、model/extrapolation risk、敏感性 |
| 可复现性与报告一致性 | 2 | analysis 可执行、轻量结果可追溯、README/JSON/API 一致、公开性合规 |

### 5.1 正式证据与实验设计（7 分）

完整分要求：

- 使用本人课程 API 产生的正式 run，并保留 experiment ID、原始/解析后配置、状态、runtime、
  validation sequence 和 final loss；
- 模型规模、训练 token 或 compute 覆盖足以支持所拟合的自由参数和最终外推；
- 每组实验有明确 hypothesis，关键对照没有同时改变过多无法归因的变量；
- `completed`、failed、partial、excluded 和 held-out run 的角色明确；
- 排除规则在看到最终结果前有合理依据，不通过事后 cherry-picking 美化曲线；
- 预算使用与最终目标一致，并讨论 exploration 与 exploitation 的权衡。

下列证据不能作为普通 completed observation：手工修改的 loss、他人的 run、私有额外训练、
只有截图没有 API provenance 的数字、失败 run 的最后一个 partial loss，以及 final result
公布后新增的探索点。

### 5.2 拟合与外推方法（7 分）

完整分要求：

- 清楚定义 target、features、units、transform 和最终预测对象；
- 说明函数形式、初始化、参数约束、objective、weighting 和 optimizer/inference 方法；
- 参数数量与有效观测量相称，避免用无法识别的高自由度模型拟合少量点；
- 报告 residual、留出预测、cross-validation、bootstrap 或 posterior predictive 等至少一种
  有效诊断；
- 对替代函数形式、run inclusion、权重或局部点做敏感性分析；
- 从拟合结果到最终配置的搜索满足 48 official accelerator-hours 约束，并考虑训练稳定性；
- 对 exploratory evaluation 到 held-out final evaluation 的差异保持明确，不把 training
  curve 拟合误写成 final-evaluation calibration。

不要求所有同学使用同一公式。较简单但可识别、诊断充分的模型可以优于复杂但无法复现或只在
训练点内表现良好的模型。

### 5.3 区间构造与风险分析（4 分）

完整分要求说明区间如何由数据和模型产生，例如 bootstrap、posterior predictive、held-out
residual calibration、模型集成或 sensitivity envelope，并解释它覆盖的随机量。仅报告
parameter confidence interval、只给 point estimate 的 standard error，或任意写“上下浮动”
而不包含 extrapolation/model-selection risk，不得获得完整分。

实际 $L$ 是否落在区间内会影响 prediction-quality 分，但 methodology 不按单次覆盖结果机械
判定。一个过程合理、事前校准但恰好 miss 的区间仍可获得方法分；一个事前没有依据但碰巧覆盖
的区间不能因此获得完整方法分。

### 5.4 可复现性与一致性（2 分）

在干净 Python 环境中，助教应能从 `results/` 运行 `analysis/`，重建主要拟合参数、最终 point
prediction 和至少两张关键图。关键数字应能回到 machine-readable row，且不依赖私有绝对
路径、未提交 notebook state 或手工编辑的中间值。

## 6. 核验流程

1. 运行 `python3 scripts/validate_repo.py`，检查目录、占位符、飞书链接、文件类型、大小和
   明显凭据。
2. 记录截止时间冻结的学生报告 commit、API final submission ID/时间和最终配置 hash。
3. 从 staff experiment export 随机抽查报告中的 experiment ID、配置、status、runtime 和
   validation loss；必要时全量比对 `results/experiments.*`。
4. 在隔离环境执行 `analysis/` 的最小复现命令，确认能从提交结果生成 fit、diagnostics、
   point prediction、interval 和图。
5. 检查 run inclusion/exclusion、单位、transform、参数约束、residual 和 sensitivity，确认
   报告结论与代码输出一致。
6. 由课程团队从冻结记录统一启动 final runs；确认 final manifest、provider terminal status、
   callback/result 与 grading export 对齐。
7. 对所有有效结果统一计算 actual-loss 指标、`E_point`、`S_interval` 和 `S_pred`；由另一名
   助教抽查公式和异常记录。

评分过程可以借助代码或模型整理材料，但实际分数和异常裁定必须由批改助教复核。自动摘要不
得覆盖原始 API/export、Git commit 或学生提交中的事实。

## 7. 异常与失败处理

| 情况 | 处理 |
| --- | --- |
| API 在截止前拒绝 final config | 该记录无效，不覆盖上一条有效 submission |
| 没有任何有效 final submission | final-loss 与 prediction 进入最低档，methodology 仍按已交材料核验 |
| 学生配置导致 final timeout/OOM/numerical failure | 不自动重跑，按无有效 actual loss 处理 |
| 确认的 backend/provider/storage 故障 | 标记 `staff_review`，审核后重跑或裁定，不直接惩罚学生 |
| README/JSON 与 API 冻结预测不一致 | API 为准；methodology 的一致性/复现性扣分 |
| exploratory run 因系统故障被退款 | 可作为故障记录，不作为完成 loss；不因平台原因机械扣实验设计分 |
| 报告代码不能运行但公式与结果可人工核对 | 按可复现性缺失扣分，不擅自重写学生实现 |
| actual loss 或预测值非有限数 | 该评分量无效，进入异常审核 |

## 8. 需要退回修正或转人工审核的情况

- GitHub 或飞书材料包含 API key、Token、Cookie、密码、私钥、内部 URL/路径或隐藏评测信息；
- 报告实验与 staff export 无法对齐，或存在手工修改、伪造、借用他人 run 的迹象；
- 把 failed/partial loss 当作 completed final loss 拟合且未说明；
- README 只有最终曲线或数字，没有 experiment provenance、分析代码或 machine-readable 结果；
- 使用最终结果反向修改原始预测、区间、拟合选择或冻结前报告；
- prediction lower/point/upper 非有限或顺序非法；
- `analysis/` 依赖未提交 notebook state、私有路径、内部数据库或无法获得的数据；
- 提交 backend、manifest、snapshot、数据、权重、完整日志、压缩包或其他明确禁止的材料；
- 攻击、抓取、逆向或绕过课程 API、预算、队列、权限或 hidden evaluation。

凭据或内部信息泄露按仓库 [`SECURITY.md`](../../SECURITY.md) 和课程安全流程立即处理；不要在
公开 Issue 或 PR 评论中复制泄露内容。
