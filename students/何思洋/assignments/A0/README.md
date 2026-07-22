# A0 公开提交：何思洋

> 本文件公开可见，仅保留脱敏摘要。服务器地址、主机名、账号、内部路径、完整命令输出、进程参数和凭据均不写入公开仓库。

## GitHub 与 PR

- 分支：`a0/ShirleYoung`
- Git 操作总结：已完成课程仓库 fork、个人仓库 clone、`upstream` 远端配置、A0 分支创建，并通过脚本生成个人学生目录。本次公开提交仅修改 `students/何思洋/` 下的作业文件。

## Linux 环境摘要

- 操作系统：Linux 5.15，x86_64
- Python：3.11.15
- Virtual environment：已使用 conda 创建用户级隔离环境
- `gpustat`：已安装在上述 conda 环境中，版本为 1.1.1
- 模拟密钥文件权限：`600`
- 常驻进程方式：`tmux`，已验证 detached session

本节不包含用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
shell reported: command not found: nvidia-smi
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

`nvidia-smi` 是 NVIDIA 驱动随附的系统命令，当前环境中找不到该命令，因此退出码为 127。`gpustat` 是用户级 Python 工具，安装成功后会通过 NVML 查询 GPU 状态；当前环境缺少可用的 NVML 共享库，所以 `gpustat` 能启动但无法查询设备，退出码为 1。

本次检查没有使用 `sudo`，没有安装或修改系统级 NVIDIA 驱动，也没有为了让命令成功而改动系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/NO95wF1gLiFNVDkreEpcEtPgn3g
- 权限：组织内持链接可查看，未开启互联网公开访问

该文档将设置为组织内公开，用于保存 A0 的组内验收材料；不会开启互联网公开访问，也不会保存 Secret、Token、Cookie、密码或私钥。

## 问题与收获

- `nvidia-smi` 不存在时应记录真实退出码和错误类别，而不是尝试安装系统驱动。
- `gpustat` 可以在用户级 conda 环境中安装，但实际查询 GPU 仍依赖系统 NVML/驱动。
- 模拟敏感配置文件应设置最小权限，例如 `600`，并且不应提交到 GitHub。
- 公开 README 只保留脱敏摘要；完整核验材料应放在组织内公开的飞书补充文档中。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 正文没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
