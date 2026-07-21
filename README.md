# OpenMOSS-暑期集训-2026

本仓库用于发布 OpenMOSS 暑期集训公开作业、接收同学的 Pull Request（PR），并维护默认 Codex 飞书机器人可以读取的公开资料索引。课程主体基于 Stanford CS336，实验室会结合内部计算资源、研究方向和评测环境发布调整后的版本。

> 重要：这是公开 GitHub 仓库。提交前请先判断材料的公开级别；不能公开的组内材料必须放在权限受控的飞书文档中。

## 信息公开规则

本项目中的材料分为三类：

| 级别 | 存放位置 | 可以包含 | 禁止包含 |
| --- | --- | --- | --- |
| 公开 | GitHub | 公开个人简介、公开代码、脱敏实验结论、公开作业说明 | 内部地址、未公开项目、原始服务器信息、组内数据 |
| 组内 | 飞书文档或 Wiki | 完整课程笔记、组内实验记录、最小必要且脱敏的审核证据、个人组内 profile | 账号密钥、App Secret、Token、Cookie、私钥 |
| 机密 | 服务器环境变量或指定密钥系统 | 机器人密钥、访问凭据、服务配置 | GitHub 和飞书文档都不得记录 |

飞书文档链接会出现在公开 GitHub 文件和 Git 历史中，因此 URL、资源标识和标题本身都属于公开元数据，不能包含保密项目名。文档正文必须设置为组织内公开，不能开启互联网公开访问；默认机器人使用的具名服务账号由实验室另行通知和授权。

## 每位同学的目录

目录名使用真实姓名，不使用 GitHub ID、拼音或昵称：

```text
students/<同学真名>/
├── PROFILE.md                  # 公开个人 profile，并登记组内个人主页链接
└── assignments/
    ├── A0/
    │   └── README.md           # 公开报告，并在其中填写组内飞书补充文档链接
    └── <A编号>/                 # A1、A2-P、A3...A6 发布后按对应题面提交
```

个人资料和每次作业都采用“双层提交”：GitHub Markdown 负责可公开、可评审、可检索的摘要；飞书文档负责组织内公开的补充材料。个人飞书主页链接写入 `PROFILE.md`，每次作业的飞书补充文档链接写入该作业的 `README.md`；仓库不使用额外的飞书索引文件。

详细边界见 [公开性与提交规则](docs/submission-rules.md)。

真实姓名目录是本集训已确定的审核规则，意味着姓名会公开并进入 Git 历史。重名或确有不能公开实名的情况必须在创建 PR 前联系课程助教处理，不要先发布再申请删除。

## PR 规则

1. Fork 本仓库，并从最新的 `upstream/main` 创建独立作业分支。
2. 每个 PR 只提交一名同学的一次作业；不要把 A0、A1 等多个作业合并到同一个 PR。
3. A0 可以创建完整的 `students/<同学真名>/`；A1 以后只修改自己的 `assignments/<作业编号>/`。Profile 更新必须单独创建 `[PROFILE]` PR。
4. 不要修改 `_template`、公共题面、其他同学的目录或仓库配置。公共文件只由助教维护。
5. PR 标题使用 `[A编号] 姓名 - 简短说明`，例如 `[A0] 张三 - 完成基础环境与 Profile`。
6. Commit 使用 Conventional Commits，例如 `feat(a0): submit 张三 environment setup`。
7. 提交前检查 git diff，删除密钥、内部地址、大型数据集、模型权重、缓存和未经脱敏的日志。
8. 未合并前直接向同一分支补交修改；PR 合并后如需修正，另开 `fix` PR。

完整命令流程与审核要求见 [公开性与提交规则](docs/submission-rules.md)。

A0 继续使用原有脚手架；A1 已发布，已有个人目录的同学使用 A1 脚手架：

```bash
python scripts/create_student.py --name '<同学真名>' --github '<GitHub ID>'
python scripts/create_assignment.py --name '<同学真名>' --assignment A1
python scripts/validate_repo.py
```

A1 还要求把 Stanford 原版仓库下载到固定兄弟目录 `../assignment1-basics`；实现和测试在
该仓库中进行，再用 `python3 scripts/sync_a1_submission.py --name '<同学真名>'` 同步
个人提交文件。完整流程见 [A1 题面](assignments/A1/README.md)。

A2-P 当前为发布候选稿，正式开放后使用同一作业脚手架：

```bash
python scripts/create_assignment.py --name '<同学真名>' --assignment A2-P
python3 scripts/sync_a2p_submission.py --name '<同学真名>'
```

上游仓库位于固定兄弟目录 `../assignment2-systems`，只同步学生自己编写的
`profiling/**/*.py`。完整要求见 [A2-P 候选题面](assignments/A2-P/README.md)。

## 作业预告

| 作业 | 主题 | 状态 | 参考资料 |
| --- | --- | --- | --- |
| [A0](assignments/A0/README.md) | Linux、GitHub、服务器环境与双层 Profile | 已发布 | 实验室原创入口作业 |
| [A1](assignments/A1/README.md) | 从零实现 tokenizer、Transformer 与训练流程 | 已发布 | [实验室题面](assignments/A1/README.md) · [Stanford 原版](https://github.com/stanford-cs336/assignment1-basics) |
| [A2-P](assignments/A2-P/README.md) | Profiling 与性能分析（A2 子作业） | 发布候选，请勿提交 | [实验室候选题面](assignments/A2-P/README.md) · [Stanford A2 固定快照](https://github.com/stanford-cs336/assignment2-systems/tree/ca8bc81a59b70516f7ebb2da4808daade877c736) |
| [A3](assignments/A3/README.md) | Scaling | 预告 | [Stanford 原版 assignment3-scaling](https://github.com/stanford-cs336/assignment3-scaling) |
| [A4](assignments/A4/README.md) | Data | 预告 | [Stanford 原版 assignment4-data](https://github.com/stanford-cs336/assignment4-data) |
| [A5](assignments/A5/README.md) | Alignment | 预告 | [Stanford 原版 assignment5-alignment](https://github.com/stanford-cs336/assignment5-alignment) |
| [A6](assignments/A6/README.md) | 内容待公布 | 预告 | 具体题目后续发布 |

A2-P 已形成发布候选稿，在状态改为“已发布”前请勿提交；A3-A5 的链接只用于
提前了解原版内容。已发布作业的提交目录、文件格式和 PR 要求以本仓库对应题面为准。

## 入口

- [作业总览](assignments/README.md)
- [同学目录说明](students/README.md)
- [贡献与 PR 说明](CONTRIBUTING.md)
- [安全与凭据泄露处置](SECURITY.md)
- [飞书机器人设置说明（可选实践）](docs/feishu-bot-setup.md)
- [默认 Codex 飞书机器人与自定义说明（可选实践）](docs/default-codex-feishu-bot.md)
