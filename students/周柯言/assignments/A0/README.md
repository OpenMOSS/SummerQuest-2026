# A0 公开提交：周柯言

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/jiqimaoke`
- Git 操作总结：已 Fork 课程仓库并 clone 到个人服务器；添加 `upstream` 为 `OpenMOSS/SummerQuest-2026`；基于 `upstream/main` 创建 `a0/jiqimaoke` 分支；使用 Conventional Commits 提交；PR 标题格式为 `[A0] 周柯言 - 完成基础环境与 Profile`。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.1 LTS
- Python：3.13.12
- Virtual environment：已创建 `venv` 并激活成功
- 模拟密钥文件权限：`-rw-------`（即 `600`）
- 常驻进程方式：`tmux`（SSH 断开后仍可保持会话运行）

> 注：公开报告中已脱敏，未包含用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

当前机器为 CPU 服务器，无 NVIDIA GPU，因此两个命令均未能成功获取 GPU 信息。

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 或驱动不可用 / 未检测到设备

### 状态解释

- `nvidia-smi` 退出码 `127` 表示系统找不到该命令，说明当前服务器未安装 NVIDIA 驱动或该工具不在 `PATH` 中。由于这是 CPU 服务器，属于预期结果。
- `gpustat` 已成功安装（版本 1.1.1），但它依赖 `nvidia-ml-py` 调用 NVML 库查询 GPU。当前机器没有 NVML 共享库，因此运行时抛出 `NVML Shared Library Not Found` 并返回退出码 `1`。
- 两个命令失败的原因不同：`nvidia-smi` 是命令层缺失，`gpustat` 是命令存在但底层 GPU 管理库无法初始化。均不需要也不应通过 `sudo` 安装系统级驱动来绕过。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/ViL0dr64loi8GuxBF1SczpwInSh

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. **区分 `origin` 与 `upstream`**：Fork 后需要将原仓库添加为 `upstream`，并在 `upstream/main` 基础上创建作业分支，避免 PR 目标错误。
2. **用户级环境管理**：使用 `python3 -m venv` 创建虚拟环境，避免使用 `sudo pip` 污染系统 Python。
3. **权限最小化**：模拟敏感配置文件通过 `chmod 600` 限制为仅所有者可读写，符合组内安全要求。
4. **常驻进程工具选择**：`tmux` 可以在 SSH 断开后保持会话，适合长时间运行的训练或日志监控。
5. **GPU 检查如实记录**：CPU 服务器上 GPU 命令失败是正常的，关键是记录退出码并解释原因，而不是为了让命令成功而修改系统环境。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。