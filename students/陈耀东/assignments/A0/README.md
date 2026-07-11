# A0 公开提交：陈耀东

## GitHub 与 PR

- 分支：`a0/chraodo`
- Git 操作总结：已 fork 课程仓库并 clone 到个人服务器；已添加课程原仓库为 `upstream`；已从主分支创建 `a0/chraodo` 分支；已使用课程脚本创建个人学生目录；使用 Conventional Commits 提交、push 到个人 fork，并向课程仓库发起 Pull Request。
- PR 标题：`[A0] 陈耀东 - 完成基础环境与 Profile`

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- CPU：x86_64 架构，Intel Xeon E5-2678 v3，多核 CPU，已通过 `lscpu` 检查
- 内存：约 251 GiB，已通过 `free -h` 检查
- 磁盘：home 目录所在文件系统空间已通过 `df -h ~` 检查，剩余空间充足
- Python：Python 3.10.12
- Virtual environment：已在用户目录下创建 Python virtual environment，`python` 与 `pip` 均来自该虚拟环境
- 模拟密钥文件权限：已创建模拟敏感配置文件，并将权限设置为 `600`，权限检查结果为 `-rw-------`
- 常驻进程方式：已了解并测试 `tmux`，可用于在 SSH 断开后保留会话或继续运行任务

本节仅保留公开、脱敏摘要，不包含用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
nvidia-smi: command not found
```

### `gpustat`

- 安装版本：gpustat 1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices.
Driver Not Loaded
```

### 状态解释

当前个人 CPU 服务器环境中未提供 `nvidia-smi` 命令，因此 `nvidia-smi` 返回 127，表示命令不存在。`gpustat` 已在用户级 Python virtual environment 中安装成功，但查询 NVIDIA 设备时返回 `Driver Not Loaded`，说明当前环境没有可用的 NVIDIA 驱动/NVML 状态。

由于个人 CPU 服务器可能没有 NVIDIA GPU，这属于可接受结果。本次检查中未使用 `sudo` 安装系统级驱动，也未修改系统级 GPU 环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/TydNwTEkHiKYNQk6wnXc1k7LnQh
- 权限状态：组织内公开，未开启互联网公开访问

该文档用于保存 A0 的组内验收材料，包括 Linux 环境检查、Python virtual environment、模拟敏感文件权限、`nvidia-smi` 与 `gpustat` 的关键结果和退出码，以及遇到的问题和排查结论。飞书文档中不保存任何 Secret、Token、Cookie、密码或私钥。

## 问题与收获

- 在服务器上执行课程脚本时发现 `python` 命令不可用，已改用 `python3`，并确认 Python 版本为 3.10.12。
- 创建 Python virtual environment 时发现系统缺少 `ensurepip` / `python3.10-venv`，未使用 `sudo` 修改系统环境，改用 `pip --user` 安装 `virtualenv` 后创建用户级环境。
- 已理解 GitHub 作业应在 `a0/chraodo` 分支完成，`origin` 用于提交个人 fork，`upstream` 用于同步课程原仓库。
- 已理解公开材料与组内材料的边界：GitHub 只保留公开、脱敏摘要；飞书补充文档保留助教核验所需的最小脱敏记录。
- 已完成 GPU 状态检查流程，并理解 `nvidia-smi` 与 `gpustat` 对 NVIDIA 驱动、NVML 和设备可见性的依赖关系。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。