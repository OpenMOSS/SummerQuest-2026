# A3 公开 Markdown 作业：龚天时

> 正式实验要求、API 字段和截止时间以课程群内助教发布的最新 A3 材料为准；GitHub 提交要求
> 见 [`assignments/A3/README.md`](../../../../assignments/A3/README.md)。本报告公开可见，
> 不得写入 API 地址或凭据、内部路径、隐藏评测信息、训练数据、权重或完整日志。

## 基本信息

- 完成范围：完成探索实验、训练协议校准、模型/数据 scaling-law 拟合、运行时间建模、不确定性分析和课程 API 最终提交。共创建 36 个实验，其中 35 个完成、1 个超时失败；另有一次配置在创建实验前被拒绝。最终 loss 拟合使用 16 个 completed 数据点。
- 未完成项：未获得最终协议下 12L/768H、256M-token 的 completed 结果；未完成 14L/16L 的最终协议迁移实验、512M 以上 token anchor 和重复随机种子实验。
- 探索预算使用情况：总预算 43,200 秒，已计费 36,331 秒（84.1%），剩余 6,869 秒（15.9%）。一次高成本运行发生基础设施超时，之后 API 入口不可用，因此没有为了耗尽预算而继续重试。

## 1. 实验设计

### 1.1 主要假设与规模变量

主要检验两个假设：在可比训练协议下，增加训练数据量 $D$ 或模型容量 $N$ 都能降低 final validation loss；在固定 48 小时最终预算下，最优选择需要平衡更大的模型和更多的训练 tokens，而不是单独最大化其中一个变量。

我主要记录两种参数量：

- $N_{\mathrm{body}}$：非 embedding 参数量，用于描述 Transformer body 容量；
- $N_{\mathrm{total}}$：API 返回的总参数量，用于训练 FLOPs 和运行时间估计。

在本次使用的模型族中，宽度与层数按 $H=64L$ 缩放，FFN intermediate size 默认为 $4H$。参数量近似为：

$$
N_{\mathrm{body}}=12LH^2+(2L+1)H,
$$

$$
N_{\mathrm{total}}=N_{\mathrm{body}}+50{,}432H.
$$

API 返回的训练 FLOPs 与 $C\approx6N_{\mathrm{total}}D$ 一致。

### 1.2 分阶段实验与控制变量

实验按以下阶段推进：

1. **流水线校准**：用小模型确认提交、轮询、结果解析和日志记录，并修正 batch-size 整除约束。
2. **初始模型/数据扫描**：模型从 2L/128H 扩展到 14L/896H，tokens 按近似对数间隔增加，确认扩大模型和数据都能降低 loss。
3. **Schedule 校准**：比较默认 cosine-to-zero 与加入 5% warmup、10% final-LR fraction 和 weight decay 0.01 的训练设置。
4. **Batch 与 LR 校准**：固定 10L/640H、32M tokens，先做 batch-2 LR sweep，再比较 batch 2/4/8/16，最后在 batch 8 附近搜索 LR。
5. **冻结协议验证**：固定 batch 8、LR 1.2e-3、AdamW/cosine、5% warmup、10% final-LR fraction 和 weight decay 0.01，扩展到 128M/256M tokens 与 12L 模型。

同一局部比较中只改变一个主要变量，保持模型族、tokenizer、sequence length、数据分布、optimizer、schedule 和 seed 不变。训练协议改变会系统性改变 loss，因此没有把全部运行强行放在一条曲线上；早期 schedule-v1 与最终冻结协议分别作为两个 protocol family，并在联合模型中使用协议相关系数。

### 1.3 优化器和超参数选择

在 6L/384H、2M tokens 上，加入 warmup、非零 final LR 和 weight decay 后，final loss 从 15.7017 降到 15.4346；单纯把 LR 降到 2e-4 反而得到 16.6792。因此后续采用完整 schedule package，而不是只降低峰值 LR。

固定 10L/640H、32M tokens、LR 8e-4 时：

| Batch | Final loss | Runtime |
|---:|---:|---:|
| 2 | 9.9993 | 679s |
| 4 | 9.9424 | 470s |
| 8 | 10.0371 | 369s |
| 16 | 10.3831 | 325s |

Batch 4 的 loss 略低，但 batch 8 明显缩短运行时间且 loss 仅退化约 0.095；batch 16 的额外加速较小、loss 则明显变差。因此 batch 8 是最终 48 小时运行更合理的时间—样本效率折中。

在 batch 8 下继续搜索 LR：

| Learning rate | Final loss |
|---:|---:|
| 8e-4 | 10.0371 |
| 1.0e-3 | 9.9263 |
| 1.2e-3 | 9.7925 |
| 1.4e-3 | 9.7822 |

LR 1.4e-3 只比 1.2e-3 改善约 0.0103，且该次运行耗时异常。该收益不足以抵消稳定性和跨规模迁移风险，因此冻结 LR 1.2e-3。

## 2. 主要结果与 Scaling-law 拟合

### 2.1 正式拟合数据

下表列出最终 loss 拟合使用的 16 个 completed 点。$N$ 表示非 embedding 参数量；协议校准、异常耗时和 failed 运行不进入该表。

| 协议 | 模型 | $N$ | Tokens | Final loss |
|---|---|---:|---:|---:|
| schedule-v1 | 6L/384H | 10.62M | 2.10M | 15.4346 |
| schedule-v1 | 8L/512H | 25.17M | 2.10M | 14.9385 |
| schedule-v1 | 8L/512H | 25.17M | 4.19M | 13.6914 |
| schedule-v1 | 8L/512H | 25.17M | 8.39M | 12.8076 |
| schedule-v1 | 8L/512H | 25.17M | 16.78M | 12.0830 |
| schedule-v1 | 10L/640H | 49.17M | 8.39M | 12.5708 |
| schedule-v1 | 10L/640H | 49.17M | 16.78M | 11.8477 |
| schedule-v1 | 10L/640H | 49.17M | 33.55M | 11.0266 |
| schedule-v1 | 10L/640H | 49.17M | 67.11M | 10.1509 |
| schedule-v1 | 10L/640H | 49.17M | 134.22M | 9.1982 |
| schedule-v1 | 12L/768H | 84.95M | 134.22M | 8.9075 |
| schedule-v1 | 14L/896H | 134.90M | 134.22M | 8.8281 |
| frozen protocol | 10L/640H | 49.17M | 33.55M | 9.7925 |
| frozen protocol | 10L/640H | 49.17M | 134.22M | 8.2095 |
| frozen protocol | 10L/640H | 49.17M | 268.44M | 7.7979 |
| frozen protocol | 12L/768H | 84.95M | 134.22M | 8.0522 |

冻结协议下，10L 模型从 32M 增加到 128M tokens 时 loss 改善 1.5830；从 128M 增加到 256M 时只改善 0.4116，显示明显的 diminishing returns。固定 128M tokens 时，10L 增大到 12L 只改善 0.1572，模型轴也开始变平。

### 2.2 拟合模型

主模型使用协议相关的容量项和数据项，同时共享 scaling exponent：

$$
\hat L_p(N,D)=E+A_p\left(\frac{N}{10^8}\right)^{-\alpha}
+B_p\left(\frac{D}{10^8}\right)^{-\beta}.
$$

普通最小二乘拟合得到共享指数：

$$
\alpha\approx0.276,\qquad \beta\approx0.126.
$$

观测尺度不足以可靠识别 irreducible loss $E$，无约束拟合把 $E$ 推到接近零。因此最终外推没有直接采信这个边界解，而是将不同 floor 假设纳入敏感性分析。

除普通联合幂律外，我还拟合了 soft-L1 robust 联合幂律、含 $\log N$、$\log D$、model-token interaction 和 protocol interaction 的线性模型，以及加入 $(\log D)^2$ 曲率项的版本：

| 模型 | RMSE | MAE |
|---|---:|---:|
| Joint power OLS | 0.1307 | 0.1153 |
| Joint power robust | 0.1312 | 0.1143 |
| Log interaction | 0.1089 | 0.0893 |
| Log interaction + quadratic token term | 0.1054 | 0.0883 |

二次模型的训练误差最低，但 16 个点不足以单凭 in-sample RMSE 支持更高阶模型。因此最终使用四模型集合与 floor 敏感性分析，而不是只选择训练误差最低的曲线。

### 2.3 Residual与leave-one-out诊断

四模型的最大 in-sample absolute residual 约为 0.20–0.24，残差均值接近零。模型集合残差没有随 $N$ 或 $D$ 单向增长，但若干中间 token 点出现约 0.17–0.19 的局部偏差，说明简单可分幂律不能完全描述 schedule 与 token curvature。

对冻结协议点分别进行 leave-one-out：

| Held-out 点 | Absolute error |
|---|---:|
| 10L，32M | 0.496 |
| 10L，128M | 0.202 |
| 10L，256M | 0.324 |
| 12L，128M | 0.155 |

最大 completed token 点被低估约 0.32，训练内拟合明显好于留一预测，说明直接沿短区间 log-linear 关系外推会过度乐观。这是最终放宽预测区间的主要依据。

### 2.4 图表说明

完整图表和逐点审计放在组织内公开的飞书补充文档中。主要图表定义与结论为：

1. **Loss–tokens 图**：横轴为 $\log_2D$，纵轴为 final validation loss，按模型和协议分组。8L与10L序列都随tokens增加而下降；冻结协议10L曲线在128M以后明显变平。
2. **Observed–fitted 图**：横轴为观测loss，纵轴为四种模型的拟合loss。训练内点接近 $y=x$，但该图不能替代leave-one-out诊断。
3. **Residual图**：横轴分别为 $\log D$ 与 $\log N$，纵轴为 $L-\hat L$。残差总体围绕零，但中间token区域存在局部结构。
4. **48小时候选图**：横轴为层数，纵轴为风险调整后的预测loss，并展示四模型范围。12–16L附近曲线很平，模型形式带宽远大于候选最低点差异。

### 2.5 失败与部分运行的处理

一次 batch-size 配置在创建实验前被拒绝，只用于修正提交约束，不作为训练数据。一次 12L/256M 运行在达到 requested runtime 后因远程训练基础设施超时失败，仅产生早期诊断值，没有 final loss。该运行：

- 不进入 completed loss fit；
- 不进入正常 runtime fit；
- 不用部分 validation loss 替代 final loss；
- 仅作为 censored runtime / failure-mode 记录。

此外，一次 completed LR 校准运行耗时相对相邻配置异常；其 loss 仅用于 LR 停止规则，runtime 不用于正常 throughput 拟合。

## 3. 最终配置与预测

- 最终配置摘要：12 layers、hidden size 768、12 attention heads；总参数量约 123.69M，其中非 embedding 参数约 84.95M；训练 10,214,309,888 tokens；sequence length 1,024；train/validation batch 为 8/2；AdamW，LR 1.2e-3，cosine schedule，warmup 0.05，final LR fraction 0.10，weight decay 0.01；16 次评估。
- 最终 loss 点预测：**5.60**。
- 预测区间：**[4.60, 6.80]**。
- 配置选择和区间构造依据：如下。

### 3.1 48小时配置选择

最终预算为 172,800 秒。纯 FLOPs 线性时间模型不能充分描述超过一百万个 optimizer steps 的固定开销，因此使用已完成冻结协议运行的 seconds/step，并采用两层保护：只分配 90% 的最终预算，同时将估计 step time 提高 15%。

| 模型 | 可训练 tokens | 四模型中位预测 | 模型预测范围 |
|---|---:|---:|---:|
| 12L/768H | 10.21B | 4.816 | 3.980–5.149 |
| 14L/896H | 8.62B | 4.774 | 4.029–5.116 |
| 16L/1024H | 7.17B | 4.763 | 4.093–5.122 |

拟合表面的最低点在14–16L，但相对12L的中位预测优势只有约0.04–0.05，远小于模型形式、LOO、LR迁移和runtime误差。12L是冻结协议下实际完成验证的最大模型，14L/16L则没有相同协议下的completed证据。因此选择12L，以很小的拟合收益换取更直接的训练证据和更可靠的runtime估计。

最终tokens对应1,246,864个optimizer steps，可被16次评估整除。按已观测12L step time估计约需135,234秒；加入15%余量后约155,519秒，低于172,800秒预算。

### 3.2 点预测与区间

最终配置上，四个直接外推模型预测约为3.98–5.15，中位数为4.82。但这些数值没有充分包含：

- 最终 $D$ 是最大 completed $D$ 的约38倍；
- power-law floor 无法稳定识别；
- 10L token 曲线已经出现明显曲率；
- 最终协议只有两个completed模型规模；
- LR只在32M–256M尺度得到迁移验证；
- 没有重复随机种子；
- 最终runtime同样属于长距离外推。

固定不同合理loss floor后，保守幂律中心约位于5.1–5.6；结合最大约0.50的leave-one-out误差和38倍token外推，我没有提交模型集合中位数4.82，而采用风险调整后的：

$$
\hat L=5.60,\qquad [L_-,L_+]=[4.60,6.80].
$$

该区间主要覆盖model-form spread、floor敏感性、LOO误差、token曲率、LR/模型规模迁移和最终超时风险。最终配置与预测曾由课程API成功确认；报告整理阶段API入口已不可用，因此以最后一次成功确认记录作为最终版本，但不声称后来观察到了显式冻结响应。

## 4. 局限性与风险

1. **协议混合**：早期数据和最终协议不同，只能用protocol indicator部分校正，不能完全消除训练动态差异。
2. **局部网格不完整**：缺少最终协议下12L/256M角点和14L/16L迁移点，model-token interaction识别较弱。
3. **远距离外推**：最终token数量约为最大completed点的38倍，是最主要的不确定性来源。
4. **Irreducible loss不可识别**：观测范围不足以稳定拟合floor，导致不同函数形式在最终尺度明显分歧。
5. **固定seed**：无法独立估计随机初始化方差，预测区间主要反映模型形式和外推风险。
6. **训练速度漂移**：远程运行存在异常耗时和超时记录，最终48小时配置仍可能因实际throughput下降而失败。
7. **超参数迁移**：batch和LR只在代表性模型与较短horizon上校准，不能证明在约10B tokens时仍是最优值。
8. **评测差异**：探索实验与隐藏最终运行共享任务定义，但隐藏运行的实际loss仍可能受到长schedule和未观测训练区域影响。

如果存在额外可靠预算，最高价值的补充实验依次是：最终协议下12L/256M、14L/128M，以及更长token anchor。这些实验分别改善局部 $N\times D$ interaction、模型大小选择和token曲率估计。

本地分析从结构化实验记录重建canonical表，按状态和协议筛选loss/runtime点，分别拟合四种模型，再在48小时runtime约束下搜索候选。按照A3公开提交规则，本PR只提交脱敏README，不提交完整日志、API响应、notebook、内部路径或后端连接代码。

## 飞书补充文档

- 链接：[龚天时 A3 Scaling Laws 补充材料](https://fudan-nlp.feishu.cn/wiki/MU8mw4BOyiudeXkzXa4cxVqanLe?from=from_copylink)

该文档设置为组织内公开，不得开启互联网公开访问；API key、Token、Cookie、密码和隐藏评测内容不得写入飞书正文。

## 自检

- [x] 本 PR 只修改我本人的 A3 目录。
- [x] 主报告为 `README.md`，所有占位符均已填写。
- [x] 报告与截止时间前的课程 API 最终提交一致。
- [x] 未提交 API 地址或凭据、内部路径、数据、权重、backend 或完整日志。
- [x] 飞书补充文档为组织内公开，且未开启互联网公开访问。
