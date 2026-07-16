# A0 公开提交：陈博闻

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/stivine`
- Git 操作总结：已 fork 课程仓库并 clone 到服务器；`origin` 指向个人 fork，`upstream` 指向课程仓库；当前分支为 `a0/stivine`。本次作业只修改 `students/陈博闻/` 目录，并通过 Conventional Commits 提交、push 到个人 fork 后向课程仓库发起 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS，Linux x86_64
- Python：Python 3.10.20
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux 3.2a

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
nvidia-smi: command not found
```

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 或驱动不可用。

```text
Error on querying NVIDIA devices.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 依赖系统中可用的 NVIDIA 驱动工具；本次执行时命令不存在，shell 对应退出码为 `127`，因此无法通过该工具查看 GPU 状态。`gpustat` 已在用户级 virtual environment 中成功安装，但它依赖 NVML 查询 NVIDIA 设备状态；当前环境缺少可用的 NVML shared library，退出码为 `1`，因此无法查询 GPU。所以当前服务器环境不能通过这两个命令获得 NVIDIA GPU 状态。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/DPRRw419LiTjmtkxxG3cNzgLnKh

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 在服务器上创建并使用了用户级 Python virtual environment，没有使用 `sudo`。
- 通过模拟敏感配置文件理解了 `600` 等权限码的含义。
- 实际执行了 `nvidia-smi` 与 `gpustat`，确认当前环境无法通过 NVIDIA 工具查询 GPU 状态。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
