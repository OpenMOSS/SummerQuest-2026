# A3：Scaling Laws——实验设计、外推与集中评测

> 状态：已发布；开放时间、冻结时间和结果发布时间以正式课程通知为准。
> 实验室题面版本：`26.2.0`（2026-08-25）。
>
> 本作业参考
> [Stanford CS336 assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)，
> 但使用实验室自建的训练 API、数据、预算与集中评测流程。上游题面中的服务地址、密钥、
> 算力环境、提交方式和评分规则不适用于本集训；冲突时以本页和课程通知为准。

A3 要求你用有限的正式实验预算研究语言模型预训练的 scaling law。你不需要重新实现训练
框架，而是通过课程 API 提交模型结构、训练规模和优化器配置，由统一的训练后端执行真实
训练并返回 validation loss。探索阶段结束后，每位同学提交一个更大预算的最终配置、一个
最终 validation loss 点预测，以及一个 80% 中心预测区间。课程团队随后统一冻结提交、运行
最终任务并在结果公布后评分。

本作业同时考察两类能力：

1. **优化能力**：在固定最终预算下，找到实际 validation loss 尽可能低的配置；
2. **推断能力**：用已有小规模实验拟合可信的 scaling law，并诚实表达外推不确定性。

评分标准和助教核验方式见 [`EVALUATION.md`](EVALUATION.md)。开始前必须阅读
[公开性与提交规则](../../docs/submission-rules.md)。本仓库公开可见，API 地址、API key、
内部服务器、挂载路径、未公开数据细节和最终评测信息不得进入 GitHub 或 Git 历史。

## 1. 学习目标

完成 A3 后，你应当能够：

1. 解释模型规模、训练 token、优化步数与训练算力之间的权衡；
2. 在有限预算下设计可识别、可比较的探索实验，而不是进行无结构的超参数搜索；
3. 拟合并比较至少一种经验 scaling-law 模型；
4. 使用 residual、敏感性分析、留出验证或其他诊断判断外推是否可信；
5. 区分点预测、预测区间、参数不确定性、模型选择不确定性和训练失败风险；
6. 提交可由实验记录和分析代码复核的最终配置与预测。

## 2. 作业阶段与当前流程

A3 有两个阶段。具体开放时间、冻结时间和结果发布时间以课程通知为准。

### 2.1 探索阶段

每位同学获得 **12 official accelerator-hours** 的正式探索预算。探索任务通过课程 API
提交，并在一致的、由课程团队控制的训练栈和硬件类型上执行。

探索阶段的目标不是简单地跑完预算，而是收集足以回答下列问题的证据：

- 在给定预算附近，模型参数量和训练 token 应如何分配？
- 哪些 learning rate、batch size、warm-up 和正则设置可以稳定训练？
- runtime、optimizer step 和 validation loss 随配置如何变化？
- 哪些实验位于最终外推区域附近，哪些只适合做稳定性检查？
- 不同拟合形式是否给出相近的最终配置和 loss 预测？

探索阶段已经结束时，不得通过额外私有训练补充“正式证据”。本作业评分只承认课程 API
记录的正式 run；本地 CPU 可以继续用于清洗结果、拟合、绘图和撰写报告。

### 2.2 最终集中评测

每位同学提交一个 **48 official accelerator-hours** 的最终配置，同时提交：

- `predicted_final_loss`：最终 validation loss 的点预测；
- `predicted_final_loss_lower`：预测区间下界；
- `predicted_final_loss_upper`：预测区间上界。

课程团队将学生的最新有效提交和报告 commit 在截止时间冻结。之后统一启动最终训练，并在
结果公布前隐藏实际 final validation loss。最终配置只执行一次；除确认的平台故障外，不因
结果不理想、超时、OOM、数值不稳定或配置选择错误而重跑。

报告、分析代码、点预测和区间必须在看到最终结果之前完成。结果公布后的修改可以作为复盘，
但不计入原始 A3 评分。

## 3. 公平性、数据与评测边界

- 课程团队保证正式探索 run 与最终 run 使用一致的训练后端和硬件类型；公开题面不承诺
  具体 accelerator 型号。
- 所有正式 run 使用课程团队固定的数据工件和 tokenized 顺序。`model_seed` 只控制模型
  初始化，不改变数据顺序。
- 探索 run 使用正式 exploratory validation evaluation；最终 run 使用更大的 held-out
  validation evaluation。二者来自相同的一般任务定义，但不是可以反复查询的同一评测。
- 训练语料来源于允许使用的公开来源，但精确 mixture、过滤、去重、tokenized shard 顺序、
  挂载路径和 final evaluation manifest 均不公开。
- 不得访问、推断或重建隐藏评测数据；不得访问他人的 API key、配置、run 或结果；不得绕过
  预算、队列、重复实验和最终提交冻结规则。

只有 API 状态为 `completed` 的 run 才有正式 final validation loss。失败 run 中的部分 loss
可以用于分析数值稳定性、显存边界或训练趋势，但不能伪装成完成结果参与同一回归目标。

## 4. 课程 API 与源码边界

课程团队在组织内发布：

- `SCALING_API_BASE_URL`；
- 每位同学独立的 `SCALING_API_KEY`；
- API quickstart；
- 可直接运行的中英文 notebook 示例；
- 参数 schema、状态和常见错误说明。

本题面冻结的公开 API contract 版本为 `26.2.0`。组织内 quickstart、notebook 和正在运行的
服务必须明确显示同一版本，并与下列认证方式、endpoint、请求和响应字段一致；若版本不一致，
停止提交并联系课程助教，不要通过猜测字段消耗正式预算。

学生只需要使用 HTTP API，不需要安装私有 client，也不需要接触计算集群 provider。最小
Python 入口使用 `requests`：

```python
import json
import os
import requests

API_BASE_URL = os.environ["SCALING_API_BASE_URL"].rstrip("/")
API_KEY = os.environ["SCALING_API_KEY"]


def api_request(method, path, **kwargs):
    response = requests.request(
        method,
        f"{API_BASE_URL}{path}",
        headers={"Authorization": f"Bearer {API_KEY}"},
        timeout=60,
        **kwargs,
    )
    try:
        body = response.json()
    except ValueError:
        body = {"raw_response": response.text}
    if not response.ok:
        raise RuntimeError(
            f"HTTP {response.status_code}: "
            + json.dumps(body, ensure_ascii=False)
        )
    return body
```

公开接口包括：

| Method | Endpoint | 用途 |
| --- | --- | --- |
| `GET` | `/budget` | 查看总预算、reserved、charged 和 remaining seconds |
| `POST` | `/submit` | 提交探索配置和 requested runtime |
| `GET` | `/experiments` | 列出本人的探索实验 |
| `GET` | `/experiment/{experiment_id}` | 查看一个实验的状态、runtime 和 validation loss |
| `POST` | `/final_submission` | 保存最终配置、点预测和预测区间 |
| `GET` | `/final_submission` | 核对当前最新有效的最终提交 |

训练配置固定包含 `model` 与 `training` 两个顶层对象。字段、默认值、取值范围和派生参数以
课程 API quickstart 为准；不要依赖未记录的字段或本地 backend 实现细节。

课程自建 backend、训练 worker 与数据工具将在项目完整归档后由维护者以独立源码仓库和固定
commit 关联到官方仓库。当前文档审核阶段不 vendoring 源码，也不创建 submodule。学生提交
不得复制、修改或替换课程 backend。

## 5. API 工作流

### 5.1 查询预算

```python
budget = api_request("GET", "/budget")
print(budget)
```

预算以秒返回。探索预算 12 小时对应 `43200` 秒。queued 和 running run 会先 reserve
完整的 `requested_runtime_seconds`；完成后按正式 accounting rule 更新 charged 和
remaining budget。

### 5.2 提交并轮询探索实验

```python
config = {
    "model": {
        "num_hidden_layers": 2,
        "hidden_size": 128,
    },
    "training": {
        "train_tokens": 4096,
        "learning_rate": 3e-4,
        "num_evals": 1,
        "model_seed": 2026,
    },
}

submitted = api_request(
    "POST",
    "/submit",
    json={
        "config": config,
        "requested_runtime_seconds": 300,
    },
)
experiment_id = submitted["experiment_id"]
```

不要照抄示例配置作为最终配置；示例只说明请求结构。提交后轮询
`GET /experiment/{experiment_id}`，直到状态进入 `completed`、`failed`、`cancelled` 或
`system_failed`。不要高频轮询；建议间隔至少 30 秒。

完成结果的 `validation_losses` 是同一个 run 在预定 evaluation cadence 上得到的 loss
序列，不是置信区间。`final_validation_loss` 是该完成 run 的最终 evaluation 结果。

### 5.3 提交最终配置与预测

```python
final_submission = api_request(
    "POST",
    "/final_submission",
    json={
        "training_config": config,
        "predicted_final_loss": 3.25,
        "predicted_final_loss_lower": 3.18,
        "predicted_final_loss_upper": 3.36,
    },
)
```

示例数字不是参考答案。最终提交要求所有预测值有限，并满足：

$$
L_- \leq \hat{L} \leq L_+.
$$

截止时间前可以重新提交；课程系统冻结最新的有效记录。提交后必须再调用
`GET /final_submission`，核对配置、点预测和上下界。无效 config 或无效区间不会覆盖上一条
有效提交。

## 6. 正式实验记录

每个探索 run 至少记录：

| 字段 | 要求 |
| --- | --- |
| `experiment_id` | API 返回的正式 ID |
| `hypothesis` | 本次 run 要回答的问题 |
| `submitted_config` | 原始 `model`、`training` 和 requested runtime |
| `resolved_config` | API 返回的参数量、optimizer steps 等派生信息 |
| `status` | `completed`、失败类别或其他终态 |
| `reserved_seconds` | API 记录的预留预算秒数，不混用墙钟等待时间 |
| `used_seconds` | API 记录的实际计费秒数，不混用墙钟等待时间 |
| `validation_losses` | 完整 loss 序列；失败 run 保留已有 sequence 或空值 |
| `final_validation_loss` | completed run 的最终 loss；失败 run 使用空值 |
| `fit_role` | 用于拟合、留出验证、稳定性诊断或明确排除 |
| `exclusion_reason` | 不参与拟合时给出可复核理由；参与拟合时保留空值 |

必须且只能选择一种格式，把公开、脱敏的轻量记录保存为 `results/experiments.csv` 或
`results/experiments.jsonl`。字段名固定为：

```text
experiment_id, hypothesis, submitted_config, resolved_config, status,
reserved_seconds, used_seconds, validation_losses, final_validation_loss,
fit_role, exclusion_reason
```

CSV 中的 config、loss sequence 等复合值使用 JSON 字符串；JSONL 每行使用包含上述字段的
一个 JSON object。没有 final loss 或排除理由时保留字段并使用空值或 `null`，不要删列。
不得提交 API key、Authorization header、API base URL、callback URL、provider job ID、
内部路径或未经裁剪的完整服务日志。

实验设计应让参数变化和 loss 变化具有可解释关系。建议在对数尺度上覆盖多个模型与数据
规模，并留出少量预算检查稳定性和外推区域。可以使用网格、分层设计、序贯设计或其他方法，
但必须解释为什么这些 run 足以识别你拟合的参数。只在一个很窄区域反复微调，或一次同时
改变过多变量而无法归因，会削弱方法有效性。

## 7. Scaling-law 拟合要求

本作业不强制唯一函数形式。可以使用 nonlinear least squares、加权回归、Bayesian
inference、robust regression、Gaussian process 或其他合理方法。一个常见起点是：

$$
L(N, D) = L_\infty + A N^{-\alpha} + B D^{-\beta},
$$

其中 $N$ 是模型规模，$D$ 是训练 token。也可以直接对 compute、optimizer steps 或其他
经过论证的变量建模。使用任何形式时，报告必须说明：

1. 因变量和每个自变量的定义、单位与变换；
2. 哪些 run 用于拟合、验证、诊断或排除；
3. 参数约束、初始化、loss function、权重和优化方法；
4. 是否处理异方差、失败/删失结果和不同 fidelity；
5. 使用 residual、留出预测、bootstrap、posterior predictive check 或其他 fit 诊断；
6. 如何从拟合模型搜索满足最终预算的配置；
7. 点预测如何从探索 evaluation 外推到 held-out final evaluation；
8. 预测区间包含哪些不确定性，以及没有包含哪些风险。

至少提交一张 scaling-law fit/extrapolation 图和一张 residual、敏感性或替代模型对照图。
图中应显示真实观测、拟合区域、最终外推点和预测区间；不要只提交没有原始点的平滑曲线。

如果多个合理模型给出明显不同的最终预测，应把 model-selection disagreement 纳入区间或
限制讨论，而不是只选择最乐观的模型。若使用 log-space 拟合，必须说明如何回到原 loss
空间，以及 back-transform bias 是否影响点预测和区间。

## 8. 80% 中心预测区间

最终区间按 **80% central prediction interval** 解释。它应覆盖“在当前实验设计、拟合方法、
最终配置和正式训练流程下，最终实际 validation loss 的预测不确定性”，而不只是回归参数的
standard error。

合理的不确定性来源包括：

- 有限探索点和观测噪声；
- scaling-law 参数估计误差；
- 拟合形式和 run 选择造成的模型不确定性；
- 从探索规模到最终规模的 extrapolation risk；
- 最终配置选择与稳定性风险；
- exploratory evaluation 与 held-out final evaluation 的差异。

过宽区间会直接受到 width penalty；过窄或偏移区间在漏掉真实 loss 时会受到更大的
miss-distance penalty。评分公式见 [`EVALUATION.md`](EVALUATION.md)。报告必须解释区间的
构造方法，例如 bootstrap quantile、posterior predictive quantile、held-out residual
calibration、模型集成或有依据的 sensitivity envelope。只写“上下浮动一个经验值”不能获得
完整方法分。

## 9. 公开提交目录

在官方仓库根目录创建个人 A3 目录：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A3
```

最终 PR 只修改：

```text
students/<同学真名>/assignments/A3/
├── README.md                         # 必交：公开 Markdown 主报告
├── {requirements.txt,pyproject.toml} # 至少一个：轻量分析依赖声明
├── analysis/                         # 必交：拟合、诊断和绘图代码
│   └── **/*.{py,md}
├── results/                          # 必交：轻量、脱敏、机器可读证据
│   ├── experiments.{csv,jsonl}       # 二选一，不能同时提交
│   ├── fit_summary.json
│   └── final_prediction.json
└── assets/                           # 必交：报告引用的压缩图
    └── *.{png,jpg,jpeg,webp,svg}
```

`analysis/` 中至少有一个可执行 `.py` 文件，并能从提交的轻量 experiment table 重建报告
中的主要拟合、诊断和图。可以在本地 notebook 中探索，但最终提交使用可执行 `.py` 文件；
不提交 notebook、notebook 导出、模型 checkpoint、训练数据或 backend 源码副本。

分析依赖使用轻量 `requirements.txt` 或 `pyproject.toml` 声明，至少提交其中一个；两者均可
提交，但任一文件不得超过 256 KiB。只声明重建分析所需的直接依赖，不提交虚拟环境、wheel、
conda environment、`uv.lock`、`poetry.lock` 或其他自动生成的大型锁文件。

`results/fit_summary.json` 至少包含：

```text
model_name, target, parameters, num_fit_runs, diagnostics, generated_at
```

其中 `parameters` 和 `diagnostics` 为 JSON object，`num_fit_runs` 为正整数。

`results/final_prediction.json` 至少包含：

```text
predicted_final_loss, predicted_final_loss_lower, predicted_final_loss_upper,
final_config 或 final_config_hash, analysis_version, generated_at
```

三个预测值必须为有限数并满足 lower ≤ point ≤ upper；`final_config` 为公开、脱敏的 JSON
object，或改用非空 `final_config_hash`。该文件必须与截止时间前 API 中的最新有效 final
submission 一致；若不一致，以冻结的 API 记录为最终配置和预测，报告会因不可复现而扣分。

## 10. README 报告要求

公开主报告必须包含：

1. 完成范围、题面版本、探索预算使用摘要和未完成项；
2. 实验设计：每组 run 的假设、控制变量、规模覆盖和预算分配；
3. 证据表：正式 experiment ID、状态、配置摘要、runtime、validation loss 和 fit role；
4. 数据清洗与纳入/排除规则，尤其是失败、部分和异常 run；
5. scaling-law 形式、参数估计方法、拟合结果和参数解释；
6. residual、留出预测、敏感性或替代模型诊断；
7. 最终配置的选择过程，以及它如何满足最终预算和稳定性边界；
8. 最终 point prediction、80% prediction interval 和区间构造方法；
9. 限制、可能失效的假设和看到真实结果前最担心的风险；
10. 最小复现命令，以及组织内公开的飞书补充文档链接。

报告中的每个关键数字必须能回到 `results/` 的一行记录或 `analysis/` 的明确计算。不得只贴
截图或手工抄写表格而不提供机器可读来源。

## 11. 文件、公开性与附件限制

| 范围 | 限制 |
| --- | ---: |
| 学生目录内任意单文件 | 不超过 5 MiB |
| A3 `README.md` | 不超过 1 MiB |
| `requirements.txt` 或 `pyproject.toml` | 每个不超过 256 KiB |
| `results/` 与 `assets/` 公开附件合计 | 不超过 2 MiB |

只允许提交：

- `README.md`；
- 根目录下的轻量 `requirements.txt` 和/或 `pyproject.toml`；
- `analysis/**/*.{py,md}`；
- `results/**/*.{csv,json,jsonl,md,txt}`；
- `assets/**/*.{png,jpg,jpeg,webp,svg}`。

明确禁止提交：

- API key、Token、Cookie、密码、私钥或包含 Authorization header 的文件；
- 内部 API URL、callback URL、服务器地址、主机名、IP、用户名、挂载路径或 provider 信息；
- 训练语料、validation data、tokenized shard、模型权重、checkpoint 或 optimizer state；
- 完整服务日志、数据库、snapshot、FrozenManifest、队列导出或其他同学的记录；
- backend/worker 源码副本、Docker image、虚拟环境、缓存、wheel、压缩包或自动生成的依赖
  锁文件；
- PDF、Office 文档、notebook 和 notebook 导出。

公开报告可以包含模型规模、训练 token、正式 experiment ID、runtime、loss、拟合参数和脱敏
图表。组织内飞书文档只保存 GitHub 不适合公开但确有审核必要的最小差量材料；不得把凭据或
隐藏评测内容放入飞书正文，也不得开启互联网公开访问。

## 12. 评分概览

总分按 100 分计：

| 部分 | 权重 | 核心问题 |
| --- | ---: | --- |
| 最终 run 的 actual validation loss | 60% | 最终配置是否在统一预算下获得较低 loss |
| 点预测与 80% prediction interval | 20% | 预测是否准确，区间是否兼顾覆盖与宽度 |
| 实验与拟合方法、报告复现性 | 20% | 证据是否有效，方法是否合理，结论是否可追溯 |

实际 final loss 越低越好。预测部分使用公开的 point-error 与 interval-score 公式。方法部分
不奖励复杂模型本身，而评估实验能否支持结论、拟合假设是否合理、诊断是否充分，以及代码能否
重建报告。完整规则见 [`EVALUATION.md`](EVALUATION.md)。

学生原因导致的 final run 超时、OOM、数值不稳定或其他配置失败，按无有效 final loss 处理；
平台或课程基础设施故障由课程团队审核、重跑或单独裁定。

## 13. 提交前自检与 PR

```bash
python3 scripts/validate_repo.py
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

一个 PR 只能修改一名同学的 `students/<同学真名>/assignments/A3/`。分支使用
`a3/<GitHub-ID>`，PR 标题使用 `[A3] 姓名 - 简短说明`，commit 使用 Conventional
Commits，例如：

```text
feat(a3): submit 张三 scaling law report
```

最终报告 commit 与 API final submission 都必须在课程截止时间前完成。课程团队将记录冻结的
报告 commit 和 API submission；不要在看到集中评测结果后重写原始预测。

## 14. 最终验收清单

- [ ] 所有正式证据都来自本人课程 API 记录，未使用他人 run 或额外私有训练冒充正式证据。
- [ ] 探索实验表包含 ID、配置、状态、runtime、loss、fit role 和排除理由。
- [ ] 完成 run、失败 run 和 partial loss 的使用方式明确且不混淆。
- [ ] 报告清楚定义了自变量、因变量、单位、变换、参数约束和拟合方法。
- [ ] 至少提供 fit/extrapolation 图和 residual、敏感性或替代模型诊断图。
- [ ] 最终配置满足 API schema，并在提交后通过 `GET /final_submission` 核对。
- [ ] 点预测和区间值有限，满足 $L_- \leq \hat{L} \leq L_+$。
- [ ] 报告解释了 80% 区间的构造、覆盖对象、已包含和未包含的不确定性。
- [ ] `analysis/` 可以从 `results/` 重建主要数字和图片。
- [ ] 已提交不超过 256 KiB 的 `requirements.txt` 或 `pyproject.toml`，并能在干净环境安装。
- [ ] `final_prediction.json`、README 与冻结前 API final submission 一致。
- [ ] 未提交凭据、内部 URL/路径、数据、权重、manifest、snapshot、backend 或大型日志。
- [ ] 文件类型、单文件大小和附件总量满足限制，飞书补充文档为组织内公开。

常用公开资料：
[Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361)、
[Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556)、
[Stanford CS336 assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling)。
