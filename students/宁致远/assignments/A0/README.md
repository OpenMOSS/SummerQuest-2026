# A0 公开提交：宁致远

> 本文件公开可见，只保留脱敏结果。不能公开但确有审核必要的最小记录将保存在组织内公开的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/zy-ning`
- Git 操作总结：已 fork 课程仓库并将个人 fork 配置为 `origin`，将课程仓库配置为 `upstream`；从主分支创建 `a0/zy-ning` 分支，使用 Conventional Commits 提交后 push 到个人 fork，并向上游主分支创建 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.1 LTS
- Python：3.12.3
- Virtual environment：已在用户 home 目录中创建并验证
- 模拟密钥文件权限：`600`
- 常驻进程方式：`nohup`（服务器当前未安装 `tmux`）

本节最终版不会包含用户名、主机名、IP、内部路径、SSH 配置或进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML/驱动不可用

### 状态解释

`nvidia-smi` 是 NVIDIA 驱动附带的系统工具，依赖正确安装的驱动与 NVML。本次执行的 exit code 为 `127`，说明当前环境找不到该命令。`gpustat` 已成功安装在 Python virtual environment 中，但它仍需要通过 NVML 读取 NVIDIA GPU 状态；本次 exit code 为 `1`，表示当前无法使用 NVML/驱动查询 GPU。这些结果只能说明当前环境无法完成 NVIDIA GPU 状态检查，不足以单独证明物理机一定没有 GPU。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/DgWvwAtXliJjdFkBhBycfCTtnHb
- 权限：组织内持链接可查看，未开启互联网公开访问

## 问题与收获

- 首次检查时 GitHub CLI 的既有登录已失效；通过 GitHub 官方网页授权重新登录后恢复了 fork、push 和 PR 流程，且没有将 Token 或其他登录凭据写入仓库。
- 公开作业只保留环境类别、软件版本、退出码和结论，完整内部记录即使位于飞书也仍需要最小化和脱敏。
- GPU 检查命令失败不等于作业失败；准确保留退出码并区分命令、设备、NVML 和驱动状态才是验收重点。

## 自检

- [x] 我已在个人 CPU 服务器实际运行 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 当前公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] 当前文档没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
