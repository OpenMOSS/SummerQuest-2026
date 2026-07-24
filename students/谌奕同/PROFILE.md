# 谌奕同 Profile

> 本文件会公开在 GitHub。只填写愿意长期公开的信息；内部项目、服务器信息、组内数据和未公开结果不要写在这里。

## 基本信息

- 姓名：谌奕同
- GitHub ID：`bondtesty`
- 常用语言：中文、英语（可阅读论文和技术文档）
- 公开身份：OpenMOSS 暑期集训 2026 学员

## 个人简介

我目前主要关注生成模型与多模态学习。日常的学习方式是读一篇新论文后，用小型合成实验验证其核心 claim，把直觉转化为可复现的代码和指标。过去一段时间围绕 flow matching、rectified flow 和 diffusion 模型做了不少 toy reproduction，也关注视觉-语言模型中的对齐与表征学习问题。

相比追逐 headline 结果，我更想知道一个方法在什么条件下成立、什么时候会失效。本次参加 OpenMOSS 暑期集训，希望借 CS336 把 tokenizer、训练循环、scaling 和 evaluation 串成一条完整的线，同时把已有的生成模型经验从“小实验直觉”推进到“可复现、可证伪的系统理解”。

## 学习与研究

### 最近关注的问题

- Flow matching 中的路径几何与条件生成：强条件信息如何降低模式混淆，schedule 和 latent geometry 对 transport 效率的影响。
- Diffusion / drifting 模型的统一视角：score-based、flow-based 和 drifting 描述在什么 toy 设定下等价，边界在哪里。
- 视觉-语言模型的表征对齐：多模态数据在时间和语义维度上的对齐问题。

### 过去探索过的方向

- 用 2D 合成数据集复现并检验了多种 flow matching 变体（conditional/unconditional、time-weighted、spherical FM 等）的 headline claim。
- 设计小型实验验证 conditioning 对生成可学习性的影响，发现强条件信息比单纯方差减少更能解决模式混淆。
- 关注多模态领域，对视觉-语言模型的结构和训练方法保持跟踪。

## CS336 学习计划

- **当前基础**：熟悉 PyTorch 和生成模型小实验，但未从头实现过完整训练循环和 tokenizer；希望补齐系统基础。
- **A1 Basics**：从头实现 tokenizer、Transformer 和训练循环，重点检查张量形状、梯度和边界条件。
- **A2 Systems**：用 profiler 分析数据吞吐和显存瓶颈，理解并行与通信。
- **A3 Scaling**：关注 scaling law 的拟合与残差分析。
- **A4 Data**：比较数据过滤、去重前后分布变化，理解质量-覆盖权衡。
- **A5 Alignment**：分析 reward hacking 和 preference optimization 的边界样例。
- **A6 Harness**：待题面发布后确定。

## 技能与工具

- 编程与框架：Python（熟练）、PyTorch（熟练）
- 工程工具：Git、Linux、Docker（基础）
- 实验工具：Weights & Biases、matplotlib、小型合成实验设计
- 其他能力：英文技术阅读、论文复现、结果可视化

## 特长、爱好与日常

- 特长：把论文里的抽象概念翻译成可运行的小实验，并记录什么情况下会失效。
- 爱好：阅读生成模型与多模态方向的最新论文，用 toy code 验证直觉。

## 公开链接

- GitHub：https://github.com/bondtesty

## 飞书文档主页

- 主页链接：https://fudan-nlp.feishu.cn/docx/Or8jdZ7DAolcxwxnT0hcpEUdnBd
- 权限状态：组织内公开

飞书主页保存更完整的个人介绍、学习记录和组内材料，不保存任何 Secret、Token、Cookie、密码或私钥。

## 公开声明

我确认本 GitHub Profile 中的信息可以长期公开展示，并可以用于 OpenMOSS-暑期集训-2026 的学习交流与作业审核。我的飞书文档主页正文设置为组织内公开。
