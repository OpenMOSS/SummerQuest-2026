# 同学目录

每位同学只创建一个以真实姓名命名的长期目录：

```text
students/<同学真名>/
```

不要使用 GitHub ID、拼音、昵称或英文缩写。A0 使用 `python scripts/create_student.py --name '<同学真名>' --github '<GitHub ID>'` 创建目录，不要直接复制或修改 `_template`。

脚手架生成的 [`PROFILE.md`](_template/PROFILE.md) 是填写模板；[`PROFILE.example.md`](_template/PROFILE.example.md) 是一份虚构的完整示例，只用于理解内容深度和写法。请描述自己的真实情况，不要复制示例中的项目、指标或链接。

模板在每个栏目都提供了 `[填写参考]`，包括初学者、工程背景和已有研究经历的写法，以及技能、特长、爱好、游戏和饮食的候选示例。填写完成后必须删除所有 `[填写参考]` 引用块；校验器会检查这一点。

A1 先把官方仓库下载到固定兄弟目录 `../assignment1-basics`，再使用
`python scripts/create_assignment.py --name '<同学真名>' --assignment A1` 创建提交目录。
实现和测试在兄弟仓库完成，随后使用
`python3 scripts/sync_a1_submission.py --name '<同学真名>'` 同步允许提交的文件。不要直接
复制或修改 `_assignment_template`，也不要提交整个原版仓库。

A1 已发布并提供专用提交模板；创建 A1 时也不要修改 `_assignment_templates`。A1 的固定
文件清单见 [A1 正式题面](../assignments/A1/README.md)。

A2-P 已提供 Profiling 专用候选模板与脚手架，但在题面状态改为“已发布”前不接收学生 PR。
正式发布后，工作仓库固定放在 `../assignment2-systems`，只通过
`scripts/sync_a2p_submission.py` 同步 `profiling/**/*.py`；trace、snapshot、上游代码和
依赖不进入个人目录。`results/` 与 `assets/` 附件合计不超过 2 MiB。详情见
[A2-P 候选题面](../assignments/A2-P/README.md)。

## 目录职责

```text
students/<同学真名>/
├── PROFILE.md                  # 公开个人 profile，并登记组内个人主页链接
└── assignments/<A编号>/
    ├── README.md               # 公开报告，并填写组内飞书补充文档链接
    └── ...                     # 代码、日志等按对应正式题面提交
```

- GitHub 中的所有文件默认可被全网访问，只能包含公开材料。
- 飞书正文设置为组织内公开，不得开启互联网公开访问；个人主页链接写入 `PROFILE.md`，每次作业直接在 `README.md` 中填写补充文档链接。
- 密钥、Token、Cookie、密码和私钥不得写入 GitHub 或飞书正文。
- 家乡、生日、MBTI、饮食、游戏和导师等个人信息均为可选项，不填写不影响审核；一旦提交到 GitHub，就应视为长期公开信息。

## PR 单位

- A0：创建完整个人目录，完成公开 profile、组内飞书 profile 和 A0 报告。
- A1：只提交自己的 A1 目录，并单独创建一个 PR。
- A2-P：正式发布后只提交自己的 A2-P 目录，并单独创建一个 PR；候选阶段不提交。
- Profile 更新：不和作业混合，单独创建 `[PROFILE]` PR。

完整要求见 [公开性与提交规则](../docs/submission-rules.md)。
