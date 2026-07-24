# A0 公开提交：俞凡

> 本文件公开可见，仅记录可长期公开的脱敏结果。服务器账号、主机名、IP、内部路径、进程参数和组内数据均不在此处保存。

## GitHub 与 PR

- 分支：`a0/Chineseyf`
- Git 操作总结：已将课程仓库 Fork 到个人 GitHub 账号，并分别配置个人 Fork 为 `origin`、课程官方仓库为 `upstream`。同步最新的 `upstream/main` 后创建独立的 A0 分支，只在 `students/俞凡/` 中完成作业。提交时将使用 Conventional Commits，并把该分支推送到个人 Fork，再向课程官方仓库的 `main` 分支发起 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.5 LTS
- Python：`3.10.12`
- Virtual environment：已在用户目录中创建并激活独立的 Python virtual environment
- 模拟敏感配置文件权限：`600`，仅当前用户可读写
- SSH 断开后继续运行进程的方式：`nohup`。它可以让程序忽略终端挂断信号，配合后台运行符号后，可在 SSH 会话断开后继续执行任务

## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在
- 关键结果：`command not found`

当前环境未提供 `nvidia-smi` 命令，因此无法通过该工具确认 NVIDIA GPU、驱动和设备状态。该结果只能说明命令不可用，不能单独证明服务器一定没有 GPU。

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 不可用
- 关键结果：`NVML Shared Library Not Found`

`gpustat` Python 包已经成功安装并能够启动，但查询 NVIDIA 设备时无法加载 NVML 共享库，因此未能返回 GPU 状态。

### 状态解释

`nvidia-smi` 是随 NVIDIA 驱动工具提供的系统命令；退出码 `127` 表明当前 shell 找不到该命令。`gpustat` 是 Python 工具，但它仍依赖 NVIDIA 的 NVML 库获取设备信息；退出码 `1` 和 NVML 错误说明 Python 命令已经运行，只是在查询后端时失败。两个结果都表示当前环境无法完成 NVIDIA 状态查询，并不能据此断言物理机器一定没有 GPU。按照作业要求，我没有为使检查成功而安装系统级 GPU 驱动或修改驱动环境。

## 飞书补充文档

- 链接：暂未有组织飞书账号

飞书补充文档仅保存助教核验所需的最小脱敏记录，包括 Linux 环境检查、两个 GPU 命令的关键结果和退出码，以及遇到的问题与排查结论。

## 问题与收获

1. 实验环境最初缺少 Git。通过检查命令是否存在，我确认问题来自基础工具缺失，而不是仓库地址或网络配置；完成基础工具准备后才继续 clone。这让我认识到，遇到命令失败时应先区分“命令不存在”“权限不足”和“远程访问失败”等不同原因。
2. 我进一步理解了 Fork、clone、`origin` 和 `upstream` 的区别：Fork 是 GitHub 上属于自己的仓库副本，clone 是某台机器上的本地副本；`origin` 指向个人 Fork，`upstream` 指向课程官方仓库。开始新作业前，应先从最新的 `upstream/main` 创建独立分支。
3. 我在不同机器的 clone 之间切换分支时遇到过本地找不到分支的问题。由此理解到，新建但尚未 push 的分支只存在于当前 clone 中，不会自动同步到另一台机器；需要在另一份 clone 中重新创建，或先 push 后再 fetch。
4. GPU 检查让我学会结合错误信息和退出码判断状态。`nvidia-smi` 不存在与 `gpustat` 无法访问 NVML 是两个层次的问题，不能简单写成“服务器没有 GPU”，更准确的结论是“当前环境无法通过相应工具完成检查”。
5. 通过 Python virtual environment 和 `chmod 600`，我理解了用户级依赖隔离与最小文件权限的重要性。Python 包应安装在独立环境中，模拟敏感配置文件只允许当前用户读写，真实凭据则不应进入 GitHub 或飞书文档。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 本公开报告已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文均已确认不包含 Secret、Token、Cookie、密码或私钥。
- [ ] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
