# 作业目录说明

A1 使用以下作业脚手架创建提交目录：

```bash
python scripts/create_assignment.py --name '<同学真名>' --assignment A1
```

A1 的原版工作仓库固定放在 `../assignment1-basics`；完成实现和官方测试后运行
`python3 scripts/sync_a1_submission.py --name '<同学真名>'` 同步个人代码。

A1 维护一个公开 `README.md`，并在其中填写组织内公开的飞书补充文档链接，同时提交
题面列出的代码、脚本和日志。

A2-P 当前是发布候选稿。正式发布后使用：

```bash
python scripts/create_assignment.py --name '<同学真名>' --assignment A2-P
python3 scripts/sync_a2p_submission.py --name '<同学真名>'
```

A2-P 的原版工作仓库固定放在 `../assignment2-systems`；同步脚本只复制
`profiling/**/*.py`。大型 trace、snapshot、上游代码和依赖不得进入本仓库；
`results/` 与 `assets/` 附件合计不超过 2 MiB。
