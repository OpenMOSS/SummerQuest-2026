# 宁致远 Profile

> 本文件会长期公开在 GitHub，只收录已公开或我愿意公开的信息。

## 基本信息

- 姓名：宁致远
- 英文名：Zhiyuan Ning
- GitHub ID：`zy-ning`
- 常用语言：中文、英语（可以阅读和撰写学术论文与技术文档）
- 公开身份：上海交通大学电子与计算机工程本科生；2026 年秋季起将在复旦大学 NLP & SII 开展博士阶段学习与研究

## 个人简介

我目前主要关注大模型的架构设计、计算系统与高效推理，兴趣横跨语言模型、多模态模型和扩散模型。过去的研究和实习经历涉及模型量化、投机解码、长上下文稀疏注意力、多语言安全评测与边缘部署。我希望通过本次集训系统补齐从 tokenizer、训练循环到分布式系统、数据和对齐的完整链路，并将研究中的经验判断转化为更可复现、可证伪的实验。

## 学习与研究

### 最近关注的问题

- 如何利用注意力和模型结构中的稀疏性，降低长上下文推理的计算和存储开销。
- 如何在不损失模型输出质量的前提下，通过量化、投机解码等方法提升推理效率。
- 如何设计可扩展、可复现的大模型实验系统，并正确分析系统与算法之间的耦合。

### 过去探索过的方向

- 语言模型与多模态大模型的推理加速、量化和系统级优化。
- 大模型多语言安全评测、安全对齐与鲁棒性。
- 边缘设备上的视觉模型、语言模型和全栈应用部署。

### 代表性项目与研究经历

#### CAS-Spec：级联自投机解码

- 背景与问题：探索无需额外 draft model 的无损大模型推理加速。
- 个人工作：参与提出并实现 Cascade Adaptive Self-Speculative Decoding 方法。
- 公开结果：工作以海报形式录用于 NeurIPS 2025。
- 公开链接：[OpenReview](https://openreview.net/pdf/7be7febdbc687ff1d863bbeaf1f37fb1b683f4bc.pdf)

#### LinguaSafe：多语言大模型安全评测

- 背景与问题：研究大模型安全对齐在多语言和对抗情境下的泛化与鲁棒性。
- 个人工作：参与多语言安全数据、评测基准与对齐实验的研究开发。
- 局限与反思：安全评测对语言、文化和攻击分布敏感，单一基准无法代表所有真实安全风险。
- 公开链接：[arXiv](https://arxiv.org/abs/2508.12733) / [GitHub](https://github.com/zy-ning/LinguaSafe)

#### Faster Than Flash：长上下文稀疏注意力解码

- 背景与问题：利用注意力稀疏性改善长上下文解码效率。
- 个人工作：作为共同作者参与研究。
- 公开结果：工作以海报形式录用于 ICML 2026。
- 公开链接：[ICML](https://icml.cc/virtual/2026/poster/61344)

## CS336 学习计划

- 当前基础：有 Python、PyTorch、模型训练与推理优化经验，希望进一步系统化数据、分布式训练、scaling 和对齐方面的知识。
- A1 Basics：从头实现核心组件，重点使用单元测试检查边界条件、张量形状和数值正确性。
- A2 Systems：使用 profiler 定位计算、内存和数据搬运瓶颈，建立性能分析的基准线。
- A3 Scaling：理解 scaling law 的建模假设，检查拟合残差和外推的可靠性。
- A4 Data：比较过滤、去重和配比前后的数据分布及其对模型行为的影响。
- A5 Alignment：系统学习偏好学习与 RLVR，通过失败样例分析 reward hacking 和评测盲点。
- A6 Harness：待题目发布后确定，优先关注可复现的评测管线与结果记录。

## 技能与工具

- 编程与框架：Python、PyTorch、FastAPI、Next.js
- 工程工具：Git、Linux，具有模型训练、推理优化和边缘部署经验
- 其他能力：学术阅读与写作、可复现实验、性能分析

## 公开链接

- 个人主页：https://zy-ning.vercel.app/
- GitHub：https://github.com/zy-ning
- Google Scholar：https://scholar.google.com/citations?user=wX_8zE4AAAAJ&hl=en

## 飞书文档主页

- 主页链接：https://fudan-nlp.feishu.cn/wiki/JucfwfiijiHBKNkQDQac3rPTnYe
- 权限状态：组织内持链接可查看，未开启互联网公开访问

## 公开声明

我确认本 GitHub Profile 中的信息可以长期公开展示，并可以用于 OpenMOSS 暑期集训 2026 的学习交流与作业审核。飞书文档主页已设置为组织内公开。
