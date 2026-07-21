# 作业总览

本集训基于 Stanford CS336。A1 已正式发布；A2-P 为发布候选稿，在状态改为
“已发布”前请勿提交。各作业的信息和状态见下表。

| 作业 | 主题 | 状态 | 原版或说明 |
| --- | --- | --- | --- |
| [A0](A0/README.md) | Linux、GitHub、服务器环境与双层 Profile | 已发布 | 实验室原创入口作业 |
| [A1](A1/README.md) | 从零实现 tokenizer、Transformer 与训练流程 | 已发布 | [实验室题面](A1/README.md) · [Stanford 原版](https://github.com/stanford-cs336/assignment1-basics) |
| [A2-P](A2-P/README.md) | Profiling 与性能分析（A2 子作业） | 发布候选，请勿提交 | [实验室候选题面](A2-P/README.md) · [Stanford A2 固定快照](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736) |
| [A3](A3/README.md) | Scaling | 预告 | [Stanford 原版 assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling) |
| [A4](A4/README.md) | Data | 预告 | [Stanford 原版 assignment4-data](https://github.com/stanford-cs336/assignment4-data) |
| [A5](A5/README.md) | Alignment | 预告 | [Stanford 原版 assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment) |
| [A6](A6/README.md) | 内容待公布 | 预告 | 具体题目后续发布 |

## 所有作业的共同提交结构

```text
students/<同学真名>/assignments/<A编号>/
└── README.md    # 公开、脱敏的报告；其他文件按各作业正式题面提交
```

A1 使用以下脚手架创建提交目录：

```bash
python scripts/create_assignment.py --name '<同学真名>' --assignment A1
```

A1 的 Stanford 工作仓库必须位于固定兄弟目录 `../assignment1-basics`；完成实现和测试后
使用 `python3 scripts/sync_a1_submission.py --name '<同学真名>'` 同步个人提交文件。

A2-P 候选题面使用固定兄弟目录 `../assignment2-systems`，但只同步
`profiling/**/*.py`；大型 profiler trace、snapshot、上游代码和依赖不进入本仓库。

课程题面、资料与个人提交都可能包含不适合公开的组内内容。请在开始任何作业前阅读 [公开性与提交规则](../docs/submission-rules.md)。
