# A0 公开提交：李哲涵

本文件公开可见，只记录脱敏结果。不能公开但确有审核必要的材料保存于组织内公开的飞书补充文档。

## GitHub 与 PR

- 分支：`a0/imadewrongfood`
- Git 操作总结：已完成课程仓库 Fork、个人 fork clone、upstream 添加、A0 分支创建，并在个人学生目录中整理本次作业材料。后续将使用 Conventional Commits 提交并发起 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.5 LTS
- Python：Python 3.10.12
- Virtual environment：已创建并使用用户级 Python virtual environment
- 模拟密钥文件权限：已设置为 `600`
- 常驻进程方式：已了解 `tmux` 可用于在 SSH 断开后保持会话继续运行

已完成当前用户、操作系统、CPU、内存和 home 目录磁盘状态检查。公开内容已删除用户名、主机名、IP、内部路径、SSH 配置、完整命令行、进程参数和组内资源规模信息。详细核验记录保存于组织内公开的飞书补充文档。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

nvidia-smi 可正常执行。公开记录中不保留主机名、GPU 型号、GPU 容量、UUID、进程信息或其他内部资源细节。

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

gpustat 可在用户级 virtual environment 中正常执行。公开记录中不保留主机名、GPU 型号、GPU 容量、UUID、进程信息或其他内部资源细节。

### 状态解释

`nvidia-smi` 成功返回，说明当前环境中的 NVIDIA 驱动与 NVML 状态可被正常访问。`gpustat` 成功返回，说明 Python 虚拟环境中的 gpustat 安装正常，并能通过 NVML 读取 GPU 状态。本次检查未使用 `sudo` 安装驱动，也未修改系统级环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/IO2gwmCV5iP2lLkd9ntcF2Kknke?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料，未开启互联网公开访问。飞书补充文档中也不保存 Secret、Token、Cookie、密码或私钥。

## 问题与收获

- 了解了公开仓库中不能提交服务器账号、主机名、IP、内部路径、进程参数、GPU 型号、GPU 容量、UUID 和组内资源细节。
- 完成了用户级 Python virtual environment 的创建，并在其中安装和运行 gpustat。
- 实际运行了 `nvidia-smi` 和 `gpustat`，记录退出码并理解二者都依赖 GPU 驱动与 NVML 状态。
- 了解了通过 `tmux` 在 SSH 断开后保持会话继续运行的基本方式。
- 配置了服务器上的 GitHub SSH key，并使用独立公钥完成 GitHub 认证。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数、GPU 型号、GPU 容量、UUID 和组内资源细节。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
