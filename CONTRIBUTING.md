# Contributing

本仓库接受两类贡献：学生作业 PR 与维护者公共文件 PR。两者不要混合。

## 学生作业 PR

1. 阅读 [公开性与提交规则](docs/submission-rules.md)。
2. A0 使用脚手架创建真实姓名目录：

   ```bash
   python scripts/create_student.py --name '<同学真名>' --github '<GitHub ID>'
   ```

3. A1 和正式开放后的 A2-P 使用作业脚手架创建提交目录：

   ```bash
   python scripts/create_assignment.py --name '<同学真名>' --assignment A1
   # A2-P 正式开放后：--assignment A2-P
   ```

   A1 开始前先把官方仓库下载到固定兄弟目录 `../assignment1-basics`。实现和测试在该目录
   中完成，每次更新后运行：

   ```bash
   python3 scripts/sync_a1_submission.py --name '<同学真名>'
   ```

4. 从最新 `upstream/main` 创建 `a0/<GitHub ID>`、`a1/<GitHub ID>` 或正式开放后的
   `a2-p/<GitHub ID>` 分支。
5. 一个 PR 只修改一个同学的一次作业。
6. GitHub `README.md` 是公开主报告，并在其中填写组织内公开的飞书补充文档链接；代码、
   日志等其他文件按对应正式题面提交。
7. 运行 `python scripts/validate_repo.py`，检查 `git diff --cached`，再 push 并创建 PR。

A1 的目录和必交文件见 [A1 正式题面](assignments/A1/README.md)。

A2-P 当前是发布候选稿，不接收学生 PR。维护者可使用固定兄弟目录
`../assignment2-systems` 测试 A2-P 脚手架和校验器；正式开放时间以
[A2-P 题面状态](assignments/A2-P/README.md)和课程通知为准。

## Profile PR

非 A0 的 profile 更新单独提交，标题使用：

```text
[PROFILE] <同学真名> - <简短说明>
```

## 维护者 PR

公共题面、模板和校验脚本由维护者修改。此类 PR 不应同时包含学生作业，并至少由另一名维护者复核影响范围与公开性。

## 安全

不要在 Issue 或 PR 中报告真实凭据、内部地址或尚未公开的研究内容。发现泄露时按 [安全与凭据泄露处置](SECURITY.md) 处理。
