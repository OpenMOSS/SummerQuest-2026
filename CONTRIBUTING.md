# Contributing

本仓库接受两类贡献：学生作业 PR 与维护者公共文件 PR。两者不要混合。

## 学生作业 PR

1. 阅读 [公开性与提交规则](docs/submission-rules.md)。
2. A0 使用脚手架创建真实姓名目录：

   ```bash
   python scripts/create_student.py --name '<同学真名>' --github '<GitHub ID>'
   ```

3. A1、A2-P、A2-K 和 A3 使用作业脚手架创建提交目录：

   ```bash
   python scripts/create_assignment.py --name '<同学真名>' --assignment A1
   # A2-P：--assignment A2-P
   # A2-K：--assignment A2-K
   # A3：--assignment A3
   ```

   A1 开始前先把官方仓库下载到固定兄弟目录 `../assignment1-basics`。实现和测试在该目录
   中完成，每次更新后运行：

   ```bash
   python3 scripts/sync_a1_submission.py --name '<同学真名>'
   ```

4. 从最新 `upstream/main` 创建 `a0/<GitHub ID>`、`a1/<GitHub ID>`、
   `a2-p/<GitHub ID>`、`a2-k/<GitHub ID>` 或 `a3/<GitHub ID>` 分支。
5. 一个 PR 只修改一个同学的一次作业。
6. GitHub `README.md` 是公开主报告，并在其中填写组织内公开的飞书补充文档链接；代码、
   日志等其他文件按对应正式题面提交。
7. 运行 `python scripts/validate_repo.py`，检查 `git diff --cached`，再 push 并创建 PR。

A1 的目录和必交文件见 [A1 正式题面](assignments/A1/README.md)。

A2-P 已正式发布，使用固定兄弟目录 `../assignment2-systems`、独立同步脚本、个人目录和
PR；完整要求以 [A2-P 题面](assignments/A2-P/README.md)和课程通知为准。

A2-K 已正式发布。它与 A2-P 共用 `../assignment2-systems`，但使用独立同步脚本、个人目录
和 PR；完整要求以 [A2-K 题面](assignments/A2-K/README.md)和课程通知为准。

A3 已正式发布。它使用课程统一训练 API，个人 PR 提交公开报告、分析代码、轻量结果、图片
和轻量依赖声明，不提交 backend、数据、权重、完整日志或 API 凭据；完整要求以
[A3 题面](assignments/A3/README.md)和课程通知为准。

## Profile PR

非 A0 的 profile 更新单独提交，标题使用：

```text
[PROFILE] <同学真名> - <简短说明>
```

## 维护者 PR

公共题面、模板和校验脚本由维护者修改。此类 PR 不应同时包含学生作业，并至少由另一名维护者复核影响范围与公开性。

## 安全

不要在 Issue 或 PR 中报告真实凭据、内部地址或尚未公开的研究内容。发现泄露时按 [安全与凭据泄露处置](SECURITY.md) 处理。
