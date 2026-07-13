# 周柯言 Profile

> 本文件会长期公开在 GitHub，只收录已公开或我愿意公开的信息。

## 基本信息

- 姓名：周柯言
- 英文名或常用名：Keyan Zhou
- GitHub ID：`jiqimaoke`
- 常用语言：中文、英语（可以阅读论文和撰写技术文档）
- 公开身份：复旦大学&上海创智学院2026级博士

## 个人简介

研究方向主要围绕大语言模型和多模态大模型的可信性，尤其是预填充阶段的事实忠实性与解码阶段的生成可靠性。过去2年参与了长上下文引用评测、多模态长文模型 detoxification 等课题，相关代码和数据集已开源。

## 学习与研究

### 最近关注的问题

- 长上下文模型在预填充阶段的事实忠实性与可验证引用。
- 大语言模型解码阶段的幻觉、安全与 detoxification 机制。
- 多模态长文档理解与评测。

### 项目与学习经历

#### MMLongCite

- 时间：2025.06 - 2025.11
- 背景与问题：构建支持长上下文细粒度引用的多模态语言模型。
- 个人工作：参与模型评测与数据集构建。
- 方法与结果：相关工作已开源，详见 `bytedance/MMLongCite`。
- 局限与反思：长上下文评测与真实应用场景仍存在差距，需要更鲁棒的指标。
- 公开链接：https://github.com/bytedance/MMLongCite

#### L-CiteEval

- 时间：2024 - 2025
- 背景与问题：长上下文引用评测，检验模型生成内容是否有原文依据。
- 个人工作：参与评测框架与数据集开源。
- 方法与结果：已发布 HuggingFace 数据集 `Jonaszky123/L-CiteEval`。
- 公开链接：https://github.com/LCM-Lab/L-CITEEVAL

#### CMD（Context-aware Model self-Detoxification）

- 时间：2024 - 2025
- 背景与问题：解码阶段的安全生成与自 detoxification。
- 个人工作：参与方法设计与实验。
- 公开链接：https://github.com/ZetangForward/CMD-Context-aware-Model-self-Detoxification

## CS336 学习计划

- 当前基础：有 PyTorch 和 Hugging Face 使用经验，熟悉长上下文与多模态评测；缺少从头训练语言模型和系统性能分析的完整经验。
- A1 Basics：重点完成 tokenizer、Transformer 组件和边界条件测试。
- A2 Systems：学习 profiler、显存分析、kernel 和并行训练，不只比较总运行时间。
- A3 Scaling：建立可复现的小规模 scaling 实验，重点检查拟合假设和误差来源。
- A4 Data：练习数据清洗、去重和质量评估，并区分公开数据与组内数据。
- A5 Alignment：理解后训练与评测流程，避免只报告单一奖励指标。
- A6 Harness：待题面发布后确定。

## 技能与工具

- 编程与框架：Python、PyTorch、Hugging Face Transformers、pytest
- 工程工具：Git、Linux、Docker 基础
- 其他能力：长上下文评测、多模态 LVLM 评测、技术写作、英文论文阅读

## 教育背景

- 本科：苏州大学，计算机科学与技术，2019 - 2023
- 硕士：苏州大学，人工智能研究院，2023 - 2026
- 博士：复旦大学&上海创智学院，2026 - 至今

## 公开链接

- 个人主页：https://jiqimaoke.github.io/
- GitHub：https://github.com/jiqimaoke
- Google Scholar：https://scholar.google.com/citations?user=NsY2j7QAAAAJ

## 飞书文档主页

- 主页链接：https://fudan-nlp.feishu.cn/docx/R2gjdQB0DokjwNxHV3gchrCQngd
- 权限状态：组织内公开

## 公开声明

我确认本 GitHub Profile 中的信息可以长期公开展示，并可以用于 OpenMOSS-暑期集训-2026 的学习交流与作业审核。我的飞书文档主页正文设置为组织内公开。
