# A0 公开提交：吴家兴

> 本文件公开可见，只记录脱敏结果。原始主机、硬件和进程信息不提交到 GitHub。

## GitHub 与 PR

- 分支：`a0/crabshellman`
- Git 操作总结：已完成课程仓库 Fork，并将个人 fork clone 到服务器；已配置 `upstream`，从最新 `upstream/main` 创建作业分支。本次作业使用 Conventional Commit 提交，并向上游 `main` 创建 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：3.11.9
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

公开报告不包含用户名、主机名、IP、内部路径、硬件容量或进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

### 状态解释

`nvidia-smi` 能够通过 NVIDIA 驱动管理接口查询设备状态，说明本次检查时命令、驱动接口和可见设备均可用。`gpustat` 安装在用户级 Python virtual environment 中，也成功通过 NVML 获取了状态。公开报告只保留退出码和状态类别，不公开主机名、GPU 型号、数量、容量、UUID、利用率或进程信息。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/FJCqde5S4oUkIvxNV67cpq1In2g

该文档设置为组织内公开，用于保存 A0 的最小脱敏验收材料，未开启互联网公开访问。

## 问题与收获

- 无头 shell 会话中没有预设 `$USER`；使用 `id -un` 获取当前用户后，继续以仅查看当前用户进程的方式完成检查。
- 学会了在用户级 virtual environment 中安装工具，未使用 `sudo pip`。
- 学会了分别记录 `nvidia-smi` 与 `gpustat` 的退出码，并区分命令成功、命令不存在、未检测到设备和 NVML/驱动不可用。
- 学会了将公开结论与私有原始证据分开保存，避免在 Git 历史中暴露内部环境信息。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
