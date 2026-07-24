# A0 公开提交：戴子皓

> 本文件公开可见，仅记录脱敏后的环境结论和退出码；组内核验所需的较详细信息保存在飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/leledaidai`
- Git 操作总结：已使用个人 fork 作为 `origin`，将课程仓库配置为 `upstream`，同步最新 `upstream/main` 后创建本作业分支，并仅在个人学生目录中完成材料。检查通过后将使用 Conventional Commits 提交，推送到个人 fork，再向课程仓库发起单人、单作业 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.2 LTS（x86_64；已省略主机名和硬件细节）
- Python：3.13.9
- Virtual environment：已在个人 home 工作区创建，并在其中安装 Python 依赖
- 模拟密钥文件权限：`600`；文件只包含明确标注的无效假值
- 常驻进程方式：`tmux`；SSH 断开前可 detach，重新登录后再 attach

CPU、内存和磁盘状态均已实际查看。为避免泄露服务器信息，公开报告不记录用户名、主机名、IP、内部路径、硬件容量、SSH 配置或完整命令行。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

### 状态解释

`nvidia-smi` 依赖 NVIDIA 驱动和设备查询接口。本次命令正常返回，说明执行检查时相关接口可用。`gpustat` 安装在个人 virtual environment 中，依赖 NVML 获取 GPU 状态；本次使用不显示进程信息的方式执行并正常返回。两条命令的原始输出可能包含服务器硬件、利用率和进程信息，因此不放入公开报告。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/PeGowuCUyirmqikLosdci3pkn7e?from=from_copylink

该文档用于保存 A0 的最小脱敏组内验收材料。文档需要设置为组织内公开，并关闭互联网公开访问。

## 问题与收获

- 初次环境核查受到执行环境限制；确认当次命令没有真正运行后，没有将其误记为作业结果，而是在环境恢复后重新实际执行并记录退出码。
- 使用 virtual environment 隔离 Python 依赖，避免 `sudo pip` 和系统 Python 污染；gpustat 最终安装并运行成功。
- GPU 检查的重点是如实执行、记录退出码并解释依赖关系，而不是强行让命令成功。本次 `nvidia-smi` 和 `gpustat` 的退出码均为 0。
- 为避免暴露其他用户进程，执行 gpustat 时不显示进程信息，公开报告也不粘贴原始 GPU 输出。
- 模拟敏感配置只使用无效假值，并通过权限检查确认文件模式为 `600`。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
