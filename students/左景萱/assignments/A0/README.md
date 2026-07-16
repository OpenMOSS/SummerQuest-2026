# A0 公开提交：左景萱

> 本文件公开可见，仅记录已核验且脱敏的结果。

## GitHub 与 PR

- 分支：`a0/for4WARD`
- Git 操作总结：已完成个人 Fork，将其配置为 `origin`，并将课程仓库配置为 `upstream`；已从最新 `upstream/main` 创建 `a0/for4WARD` 分支，使用 Conventional Commits 提交并 push 到个人 Fork，随后向课程仓库 `main` 创建 Pull Request。本次仅修改个人学生目录。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04 LTS
- Python：3.9.19
- Virtual environment：已在用户 home 目录创建并验证可用，未修改系统 Python
- 模拟密钥文件权限：`600`，仅当前用户可读写
- SSH 断开后继续运行：可使用 `nohup command > task.log 2>&1 &`，再通过日志和进程状态检查任务

公开摘要未包含用户名、主机名、IP、内部路径、硬件容量或进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功
- 脱敏摘要：系统命令正常返回。公开报告不记录 GPU 型号、数量、UUID、利用率或进程信息。

### `gpustat`

- 安装版本：1.1.1
- 安装命令 Exit code：0
- 运行命令 Exit code：0
- 状态类别：成功
- 脱敏摘要：在用户 virtual environment 中从官方 PyPI 安装成功；使用 `--no-processes` 查询并正常返回，公开报告不记录设备明细。

### 状态解释

`nvidia-smi` 成功说明当前环境能够调用 NVIDIA 管理接口；这不代表本次 A0 申请或使用了 GPU 计算资源。`gpustat` 是依赖 Python 包及 NVML 的上层工具；在用户 virtual environment 中安装后也正常返回。两项检查均只记录退出码和状态类别。未使用 `sudo`，也没有安装驱动、CUDA 或修改系统库。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/D9BWdjdaho3mh6xA1ZqccCgFnvd
- 权限状态：组织内公开，未开启互联网公开访问

## 问题与收获

- 提示词最初把 GPU 查询限定为人工步骤；经授权后改为代理可直接执行，并要求原始输出只用于本地核验、提交前脱敏。
- 个人 Fork 创建后，通过代理核验其默认分支与上游基线一致，并确认本地 `origin`、`upstream` 分工正确。
- 默认包源最初没有匹配的 `gpustat` 发行包；经本人授权后在隔离的 virtual environment 中临时指定官方 PyPI，安装与查询均成功。
- 通过 virtual environment 隔离 Python 包，并用 `600` 权限完成模拟敏感文件练习。

## 自检

- ✅ 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- ✅ 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- ✅ 公开内容已删除用户名、主机名、IP、内部路径、硬件容量、进程参数和组内数据。
- ✅ GitHub 公开文件没有 Secret、Token、Cookie、密码或私钥。
- ✅ 飞书补充文档已设置为组织内公开，且未开启互联网公开访问。
