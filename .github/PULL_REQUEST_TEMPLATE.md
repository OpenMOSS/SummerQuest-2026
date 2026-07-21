# PR Checklist

## 基本信息

- 同学真名：
- 作业编号：A0

## 修改范围

- [ ] 本 PR 只包含我本人本次作业的文件。
- [ ] 我没有修改其他同学、`students/_template`、公共题面或仓库配置。
- [ ] PR 标题符合 `[A编号] 姓名 - 简短说明`。

## 公开性与安全

- [ ] 我理解 GitHub 中所有内容均为公开资料。
- [ ] 公开内容不含内部主机名、IP、账号、路径、数据或未公开项目。
- [ ] GitHub 和飞书正文均不包含 Secret、Token、Cookie、密码或私钥。
- [ ] 我检查了 `git diff --cached`，没有提交大型数据、模型权重、缓存或完整日志。

## 双层提交

- [ ] `README.md` 提供可公开、已脱敏的作业报告。
- [ ] `README.md` 已填写飞书补充文档链接。

## A1 额外检查

若本 PR 不是 A1，请将本节标记为不适用。

- [ ] 已提交 `README.md`、`submission/cs336_basics/`、`submission/tests/adapters.py`、
      `submission/scripts/` 和 `logs/`。
- [ ] 报告文件为 `README.md`（Markdown）。
- [ ] `assignment1-basics` 工作仓库位于 SummerQuest 的兄弟目录，且没有被提交到本 PR。
- [ ] 已在 `../assignment1-basics` 运行官方测试，并使用同步脚本更新个人提交目录。

## A0 额外检查

若本 PR 不是 A0，请将本节标记为不适用。

- [ ] 已完成公开 GitHub profile 和组内飞书 profile。
- [ ] 已在个人服务器实际运行 `nvidia-smi` 与 `gpustat` 并记录退出码。
- [ ] A0 的组内验收材料已放入 README 链接的飞书补充文档。

## A2-P 额外检查

若本 PR 不是 A2-P，请将本节标记为不适用。A2-P 在题面状态改为“已发布”前不接收学生
PR。

- [ ] 已提交 Markdown `README.md`、`submission/profiling/**/*.py`、规定的轻量
      `results/` 和至少三张被报告引用的裁剪、压缩图片。
- [ ] 已运行 `scripts/sync_a2p_submission.py` 和 `scripts/validate_repo.py`。
- [ ] 已使用 nsys 或 `torch.profiler` 完成两个模型规模 × 三个 context 的六个
      `train_step` trace。
- [ ] 仓库外源码与资料引用使用固定 commit 的 GitHub HTTPS 绝对 URL，未使用本机或跨仓库相对链接。
- [ ] 未提交 `.nsys-rep`、snapshot、完整 trace、权重、数据、压缩包、上游代码或依赖环境。
- [ ] `results/` 与 `assets/` 合计不超过 2 MiB，Profile 只保留关键汇总与截图。

## 给助教的说明

<说明未完成项、环境限制或希望重点审核的内容。不要粘贴密钥和内部地址。>
