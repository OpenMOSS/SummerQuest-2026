# A3：Scaling Laws——Markdown 作业提交

> 状态：已发布；截止时间和结果发布时间以课程通知为准。
>
> A3 题面、API Quickstart V2 和工作流 notebook 已由助教在课程群内发布。实验配置、API
> 字段、预算和最终提交流程以群内最新版本为准。本页只说明 GitHub 作业提交方式，不重复
> 发布群内材料。

## 提交内容

A3 的 GitHub PR **只要求一份公开、脱敏的 Markdown 作业报告**：

```text
students/<同学真名>/assignments/A3/
└── README.md
```

报告至少说明：

1. 完成范围、未完成项和探索预算使用情况；
2. 实验设计、主要配置与选择这些实验的理由；
3. 主要结果、scaling-law 拟合方法和必要的诊断；
4. 最终配置、最终 loss 点预测和预测区间；
5. 局限性、失败实验的处理和对外推风险的判断；
6. 组织内公开的飞书补充文档链接。

公式、表格、代码片段和图表说明均写入 `README.md`，不提交 PDF、Office 文档或 notebook
导出文件。报告应让助教能够理解实验依据和最终判断，但不要求在公开仓库中复刻课程后端。

如果需要说明本地分析环境，可以在同目录附上不超过 256 KiB 的轻量
`requirements.txt` 或 `pyproject.toml`；两者都是**可选项，不是必交项**。不要提交 lock
file、虚拟环境、wheel、缓存或完整依赖环境。

## 创建目录

从最新的 `upstream/main` 创建个人 A3 分支后，在仓库根目录运行：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A3
```

填写生成的 `README.md`，删除所有占位符，并在提交前运行：

```bash
python3 scripts/validate_repo.py
git status --short
git diff --check
git diff --cached
```

## PR 规则

- 一个 PR 只修改一名同学的 `students/<同学真名>/assignments/A3/`；
- 分支名使用 `a3/<GitHub-ID>`；
- PR 标题使用 `[A3] 姓名 - 简短说明`；
- commit 使用 Conventional Commits，例如
  `feat(a3): submit 张三 scaling law report`；
- 截止时间前完成 GitHub 报告和课程 API 的最终提交；看到集中评测结果后不要回改原始预测。

## 公开性

本仓库公开可见。不得提交 API 地址或 key、Authorization header、内部服务器或路径、训练
数据、隐藏评测信息、模型权重、完整日志、其他同学的记录或课程 backend。确有审核必要但
不适合公开的最小差量材料放入组织内公开的飞书补充文档；凭据和隐藏评测内容仍不得写入飞书。

开始前还应阅读[公开性与提交规则](../../docs/submission-rules.md)。
