# A0 公开提交：吴睿曦

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/hetiankong`
- Git 操作总结：使用 GitHub CLI 完成授权并创建个人 fork，将个人 fork 配置为 `origin`、官方仓库配置为 `upstream`；同步最新 `upstream/main` 后创建 `a0/hetiankong` 分支，在该分支完成报告、提交并向官方仓库发起 PR。

## Linux 环境摘要

- 操作系统：Linux
- Python：3.13.11
- Virtual environment：已创建并验证
- 模拟密钥文件权限：600
- 常驻进程方式：`tmux`

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：命令执行成功；本机没有 GPU

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 库不可用

### 状态解释

本机没有 GPU。`nvidia-smi` 命令本身能够正常执行，退出码为 0；这只说明系统工具可用，不代表本机存在可用 GPU。`gpustat` 通过 Python 的 NVML 绑定读取 GPU 状态，当前环境无法加载 NVML 库，因此退出码为 1。我没有为了让检查成功而安装驱动或修改系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/XbwFwj3WGidi2FkbAj7c8OQFnug

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 我理解了 GitHub fork、`origin` 与 `upstream` 的分工，并学会从最新的官方主分支创建独立作业分支。
- 我学会了使用虚拟环境隔离 Python 依赖，并通过 `600` 权限限制模拟敏感文件只能由当前用户读写。
- 在 SSH 场景中，我实际练习了使用 `tmux` 创建和恢复脱离会话，理解了让长时间任务不依赖 SSH 连接持续运行的基本思路，也了解到 `systemd --user` 和 supervisord 等其他可选方案。
- 本机没有 GPU，但 GPU 检查仍需要分别记录命令退出码并解释状态，而不是为了得到成功输出而修改系统驱动。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
