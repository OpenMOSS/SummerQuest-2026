# 公开性与提交规则

本文档是所有作业的共同规则。各作业题面可以增加要求，但不能降低这里的保密和凭据安全标准。

## 1. 三类材料

### 公开材料：提交到 GitHub

可以包含公开个人简介、自己编写且允许公开的代码、脱敏后的命令输出、公开数据上的实验结果和公开参考资料。

不得包含内部主机名、IP、账号、目录结构、未公开项目名、组内数据样例、内部评测集、内部聊天内容或尚未公开的研究结果。

### 组内材料：提交到飞书

每位同学维护一个组织内公开的个人飞书主页；每次作业再建立一份组织内公开的作业补充文档。不能公开到互联网但可以在组织内共享，且确有审核必要的脱敏证据、内部资源使用记录和课程笔记放在对应飞书文档中。飞书不是第二份完整报告：GitHub `README.md` 是公开主报告，飞书只保存组织内差量。

飞书文档必须：

- 关闭互联网公开访问。
- 设置为组织内公开；默认机器人使用的具名服务账号按实验室通知单独授权。
- 个人主页链接在 `PROFILE.md` 中登记；每次作业在该作业 `README.md` 中填写飞书补充文档链接。
- 避免在标题中使用不能公开的项目代号。飞书 URL、资源标识和标题会出现在公开仓库与 Git 历史中，本身即是公开元数据。

### 机密材料：不提交

App Secret、Verification Token、Encrypt Key、Webhook Secret、Access Token、Refresh Token、Cookie、SSH 私钥、密码和数据访问凭据只能保存在服务器环境变量或指定密钥系统中。

`.env` 被 `.gitignore` 忽略不代表绝对安全。提交前仍需执行 `git status` 和 `git diff --cached`。

## 2. 双层 Profile 与双层作业

个人资料：

- `PROFILE.md`：公开版本，并登记个人组内主页链接和权限状态；不登记飞书机器人信息。

每次作业：

- `assignments/<A编号>/README.md`：公开、脱敏的报告和公开代码说明，并在“飞书补充文档”一节填写组内文档链接。
- 代码、日志、图表等其他提交文件：仅在对应正式题面规定的个人作业目录内提交。A1 的
  固定目录和必交文件见 [A1 题面](../assignments/A1/README.md)。

公开 README 应该让评审者理解你完成了什么，但不要求通过公开仓库重建实验室内部环境。组内飞书文档只保存不能公开且审核必要的差量证据，不要机械复制公开报告，也不要把组内正文复制回 GitHub。仓库不维护额外的飞书索引文件。

## 3. 一次作业的标准 PR 流程

首次准备 fork：

```bash
git clone https://github.com/<你的 GitHub ID>/SummerQuest-2026.git
cd SummerQuest-2026
git remote add upstream https://github.com/<实验室组织>/SummerQuest-2026.git
git fetch upstream
```

每次作业从最新主分支新建分支：

```bash
git switch main
git fetch upstream
git rebase upstream/main
git push origin main
git switch -c a0/<你的 GitHub ID>
```

A1 切换到对应分支并创建作业目录：

```bash
python scripts/create_assignment.py --name '<同学真名>' --assignment A1
```

A1 的官方工作仓库必须位于固定兄弟目录 `../assignment1-basics`。在该目录完成实现和
`uv run pytest` 后，回到 SummerQuest 仓库同步允许提交的文件：

```bash
python3 scripts/sync_a1_submission.py --name '<同学真名>'
```

不要把整个 `assignment1-basics/`、公共 tests/fixtures、数据、模型权重或依赖环境放进
SummerQuest 仓库。

完成后只暂存自己的作业目录，检查差异再提交。A0 使用第一条 `git add`；A1 使用第二条：

```bash
git add "students/<同学真名>"
# git add "students/<同学真名>/assignments/A1"
git status
git diff --check
git diff --cached
git commit -m "feat(a0): submit <同学真名> environment setup"
git push -u origin a0/<你的 GitHub ID>
```

在 GitHub 创建指向上游 `main` 的 PR。标题使用：

```text
[A0] <同学真名> - 完成基础环境与 Profile
```

A1 将分支名、commit scope 和标题中的编号改为 `a1` 和 `[A1]`。

## 4. PR 范围

- 一个 PR 只包含一名同学的一次作业。
- A0 可以创建自己的完整目录；之后只修改自己的对应作业目录。
- 不修改其他同学、`students/_template`、`assignments/` 公共题面或仓库配置。
- 不提交数据集、模型权重、虚拟环境、缓存、完整运行日志或其他大文件。
- 同一个 PR 收到反馈后继续 push 修复，不新开重复 PR。
- 作业 PR 已合并后需要修正时，另开 `[A编号][FIX]` PR。

Profile 更新不与作业混合，单独使用 `[PROFILE] <同学真名> - <说明>` PR。

## 5. 提交前检查

```bash
git status --short
git diff --check
git diff --cached --stat
git diff --cached
```

逐项确认：

- GitHub 内容可以被任何互联网用户看到。
- 飞书正文对组织内成员可见；机器人服务账号只按实验室明确通知授权。
- 所有主机名、IP、用户名、内部路径和内部数据都已从公开材料中删除或脱敏。
- Git 历史中没有曾经提交过密钥；仅在后续删除文件不能消除泄露。
- 飞书链接可由组织内成员打开，但没有设置为互联网公开。

## 6. 凭据泄露处置

如果凭据曾进入 commit、PR、飞书正文、截图或日志：

1. 立即停用或轮换凭据，不等待删除文件。
2. 通知课程助教，说明暴露位置和时间。
3. 清理 Git 历史或飞书版本记录；仅创建一个删除 commit 不足以消除泄露。
4. 验证旧凭据已经失效，再恢复服务。
