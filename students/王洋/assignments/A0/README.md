# A0 公开提交：王洋

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/piacesy`
- Git 操作总结：已完成课程仓库 Fork，并配置个人 Fork 为 `origin`、课程仓库为 `upstream`；从最新 `upstream/main` 创建 `a0/piacesy` 分支，使用 Conventional Commit 提交并推送到个人 Fork，并已向上游 `main` 创建 [PR #38](https://github.com/OpenMOSS/SummerQuest-2026/pull/38)。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04 LTS
- Python：3.10.12
- Virtual environment：已使用用户级 `virtualenv` 创建
- 模拟密钥文件权限：600
- 常驻进程方式：`tmux`，已验证 detached session 可正常创建和运行

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

### 状态解释

当前环境中没有 `nvidia-smi` 可执行文件，因此 shell 返回 127，属于命令不存在，而不能仅凭该结果断言硬件状态。`gpustat` 已在用户级 virtual environment 中成功安装，但查询设备时需要 NVIDIA 驱动提供的 NVML 共享库；当前环境找不到 NVML，因此查询失败并返回 1。两项结果符合 CPU 服务器可能没有 NVIDIA 驱动环境的情况，检查过程中未安装系统级驱动或修改系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/NTTgwkLoiiult0kh0WucE58Bn6d

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 系统 Python 缺少 `ensurepip`，无法直接使用标准库 `venv`；改为在用户级安装并使用 `virtualenv`，未使用 `sudo pip`。
- 创建了模拟敏感配置文件并将权限设置为 600，理解了最小权限原则。
- 使用 `tmux` 创建 detached session，验证了 SSH 断开后继续运行任务的基本方式。
- 分别保存两个 GPU 检查命令的退出码，并区分了“命令不存在”和“Python 工具已安装但 NVML 不可用”。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
