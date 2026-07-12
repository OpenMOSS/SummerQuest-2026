# A0 公开提交：李畅松

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/maybe1possible`
- Git 操作总结：已在本地 clone 个人 fork，添加实验室仓库为 `upstream`，执行 `git fetch upstream` 获取最新主分支，并从 `upstream/main` 创建当前 A0 分支。当前提交范围限制在 `students/李畅松/` 目录内；后续完成飞书链接补充后，将使用 Conventional Commits 提交并向上游仓库创建 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.5 LTS，Linux x86_64。
- Python：Python 3.13.5。
- Virtual environment：已创建用户级 Python virtual environment，并在其中安装 `gpustat`。
- 模拟密钥文件权限：已创建模拟配置文件并设置权限为 `600`。
- 常驻进程方式：了解并选择 `tmux` 作为 SSH 断开后继续运行交互式任务的方式；也可按服务化需求使用 `systemd --user`。

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
nvidia-smi: command not found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 是 NVIDIA 驱动工具，本环境中该命令不存在，因此返回 127。`gpustat` 已在用户级 virtual environment 中成功安装并执行，但它需要通过 NVML 查询 NVIDIA 设备状态；当前环境没有可用的 NVML shared library，因此返回 1。该结果说明当前环境无法通过 NVIDIA 工具检查 GPU 状态，不代表需要或应该自行安装系统级驱动。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/QIqAwARP5iWf7qkIW4hcozFcnnd

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 配置了 `upstream` 并从上游最新主分支创建作业分支，确认 A0 修改范围只在个人目录内。
- 完成了 Linux 基础信息、Python virtual environment、模拟密钥文件权限和当前用户进程查看等基础操作。
- 实际运行了 `nvidia-smi` 和 `gpustat`，并记录退出码；本环境中 `nvidia-smi` 不存在，`gpustat` 因 NVML 不可用而失败。
- 公开报告只保留脱敏摘要，不记录用户名、主机名、IP、内部路径、完整进程参数或凭据。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
