# A0 公开提交：李航成

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/lkdhy`
- Git 操作总结：已 Fork 课程仓库，clone 到个人服务器，添加课程仓库为 `upstream`，并从 `upstream/main` 创建 `a0/lkdhy` 分支。后续将在该分支提交 A0 作业并创建 PR。

## Linux 环境摘要

- 操作系统：Linux x86_64，kernel 5.15
- Python：Python 3.12.0
- Virtual environment：已在仓库外创建并激活 Python virtual environment
- 模拟密钥文件权限：600
- 常驻进程方式：tmux 3.2a，可在 SSH 断开后保留会话并继续运行命令

本次检查确认机器约有 128 个逻辑 CPU、约 1.5 TiB RAM，根文件系统约 7.0T、使用率约 44%。公开报告中已省略用户名、主机名、IP、内部路径和完整挂载细节。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
bash: nvidia-smi: command not found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

当前环境是 CPU 机器，没有可用的 `nvidia-smi` 命令，因此 `nvidia-smi` 返回 127。`gpustat` 已在 virtual environment 中安装并可以启动，但它依赖 NVIDIA NVML 查询 GPU 状态；当前环境找不到 NVML shared library，因此返回 1。根据 A0 要求，本项重点是如实执行、记录退出码并解释状态，不需要安装系统级驱动或修改系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/NrhCws3wEiPcFokaQgjcYrmQnDf

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- CPU 机器上没有 `nvidia-smi` 命令，返回码为 127；这说明命令本身不可用，而不是具体 GPU 负载异常。
- `gpustat` 可以在 virtual environment 中安装和启动，但依赖 NVIDIA NVML；当前环境缺少 NVML shared library，因此无法查询 GPU 状态。
- 后续提交公开材料时，需要只保留脱敏摘要和关键退出码，避免提交用户名、主机名、IP、内部路径、进程参数或包源日志。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
