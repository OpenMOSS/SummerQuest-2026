# A0 公开提交：杨永卓

## GitHub 与 PR

- 分支：`a0/yatao-zhuozhuo`
- Git 操作总结：已从个人 fork 创建本地仓库，添加实验室仓库为 `upstream`，从最新 `upstream/main` 创建 `a0/yatao-zhuozhuo` 分支，并只在 `students/杨永卓/` 目录下完成 A0 作业内容。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.3 LTS (Jammy Jellyfish)，x86_64 架构
- CPU：Intel Xeon Gold 6530，2 sockets，每 socket 32 cores，每 core 2 threads
- 内存：约 1.0 TiB，总体可用约 870 GiB；未配置 swap
- 磁盘：当前工作区所在文件系统约 3.5 TiB，已使用约 59%
- Python：Python 3.12.4
- Virtual environment：已在个人 home 目录下的 A0 工作区创建并启用 `.venv`，`python` 和 `pip` 均来自该虚拟环境
- 模拟密钥文件权限：`mock_config.env` 权限为 `600`
- 常驻进程方式：已验证 `tmux 3.0a` 可用；可以在 tmux 会话中启动长任务，通过 detach 退出 SSH 前台，之后再 attach 回到同一会话继续查看进程状态

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
NVIDIA-SMI 580.95.05
Driver Version: 580.95.05
CUDA Version: 13.0
GPU 0: NVIDIA
Memory: 51 MiB / 49140 MiB
GPU-Util: 0%
Processes: No running processes found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
[0] NVIDIA | 30C, 0 % | 50 / 49140 MB
```

### 状态解释

`nvidia-smi` 能够正常运行并返回 exit code 0，说明当前环境中 NVIDIA 驱动、NVML 和命令行工具可用，并且系统能够识别到一张 NVIDIA 显卡。输出中没有正在运行的 GPU 进程，显存仅有少量基础占用，GPU 利用率为 0%，说明检查时没有明显计算任务占用该卡。

`gpustat` 安装在用户级 Python virtual environment 中，版本为 1.1.1，运行后同样返回 exit code 0。它依赖 Python 包和 NVML 读取 GPU 状态，因此在 `nvidia-smi` 可用、驱动和 NVML 正常的情况下也能成功展示 GPU 型号、温度、利用率和显存占用。没有使用 `sudo pip`，也没有为了检查 GPU 修改系统级驱动或系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/GNPhwhMqXip6KwkzDfzcyiPWnqh?from=from_copylink
- 权限状态：组织内公开，未开启互联网公开访问

该文档用于保存 A0 的组内验收材料。

## 问题与收获

- 通过查看操作系统、CPU、内存和磁盘状态，确认了提交材料中只需要保留脱敏后的资源摘要，不应公开用户名、主机名、内部路径或完整进程参数。
- Python virtual environment 创建成功后，`python` 和 `pip` 都指向 `.venv` 内部路径，说明后续 Python 包安装可以限制在用户级环境中，避免污染系统 Python。
- 模拟敏感配置文件使用示例值而不写真实凭据，并将权限设置为 `600`，可以减少配置文件被其他用户读取的风险。
- `tmux` 的 detach/attach 流程可以让长任务在 SSH 断开后继续运行，适合训练、日志观察等需要较长时间保持的任务。
- `nvidia-smi` 和 `gpustat` 都成功运行，说明当前服务器能够通过系统工具和 Python 工具两种方式读取 GPU 状态；两者都依赖 NVIDIA 驱动和 NVML，但 `gpustat` 还依赖虚拟环境中的 Python 包。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
