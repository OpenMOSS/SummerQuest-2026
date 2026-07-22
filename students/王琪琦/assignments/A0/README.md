# A0 公开提交：王琪琦

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/Lilyiooo`
- Git 操作总结：已 Fork 课程仓库并 Clone 到个人服务器，已将实验室仓库配置为 `upstream`，并从最新的 `upstream/main` 创建 `a0/Lilyiooo` 分支。本次作业使用 Conventional Commits 提交，分支推送到个人 Fork，并向课程仓库提交 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.1 LTS
- Python：Python 3.10.16
- Virtual environment：已激活用户级 Conda 环境 `fourier-llama`
- 模拟密钥文件权限：600，仅当前用户可读写
- 常驻进程方式：`tmux`

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：命令执行成功，但未返回可核验的设备信息

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

```text
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 命令可以执行并返回退出码 0，但没有输出可核验的设备状态，因此不能据此认定当前有可用 GPU。`gpustat` 已安装在当前 Conda 环境中，但运行时无法加载 NVML 共享库，因此无法通过 NVML 查询 NVIDIA GPU 状态。两个工具都依赖系统提供的 NVIDIA 驱动和 NVML；本次检查没有使用 `sudo` 安装驱动或修改系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/XlDpwYkvZiD3Nvkas0ec3kMqnKf?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 现有 Conda 环境也可以用于用户级 Python 包管理。通过 `python -m pip` 安装包，可以确保包被安装到当前激活的 Python 环境。
- 系统路径中存在某个命令，不代表当前 Conda 环境已安装对应的 Python 包；我使用 `python -m pip show gpustat` 确认安装状态。
- 退出码需要和实际输出一起分析。本次 `nvidia-smi` 返回 0 却没有设备输出，而 `gpustat` 明确报告 NVML 不可用，因此不能只根据单个退出码判断 GPU 状态。
- 模拟密钥文件设置为 `600` 后，只有当前用户可读写，这是存放敏感配置时应遵循的最小权限原则。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
