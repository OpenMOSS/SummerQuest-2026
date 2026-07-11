# A0 公开提交：雷梓一

> 本文件公开可见，仅记录经过脱敏的结果。内部服务器信息、账号、路径、硬件容量、完整日志和组内材料均未写入本文件。

## GitHub 与 PR

- 分支：`a0/lazyy11`
- Git 操作总结：已完成课程仓库 Fork，在个人服务器中 clone 个人 Fork，并将实验室原仓库配置为 `upstream`；随后同步最新 `upstream/main`，创建独立分支 `a0/lazyy11`，使用脚手架生成个人目录。本次作业只修改个人学生目录，并使用 Conventional Commits 提交、推送分支后向上游 `main` 创建 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 20.04.6 LTS，x86_64
- Python：Python 3.14.6
- Virtual environment：已创建用户级 Python virtual environment
- 模拟密钥文件权限：600
- 常驻进程方式：`nohup`

公开摘要未包含用户名、主机名、IP、内部路径、CPU 与内存容量或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
bash: nvidia-smi: command not found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML、驱动或设备访问状态不可用，当前无法查询 NVIDIA 设备

```text
Error on querying NVIDIA devices.
Unknown Error
```

### 状态解释

`nvidia-smi` 的退出码为 127，且 Shell 明确提示命令不存在，说明当前环境无法找到 `nvidia-smi`。这一结果只能说明当前用户环境无法通过该命令检查 NVIDIA GPU，不能据此断定物理服务器一定没有 GPU。

`gpustat` 已成功安装在用户级 virtual environment 中，但运行时返回退出码 1，并提示无法查询 NVIDIA 设备。`gpustat` 依赖底层 NVIDIA 驱动、NVML 和设备访问，因此当前只能判断底层查询条件不可用。现有输出不足以进一步区分物理机器无 GPU、驱动未安装、NVML 不可用或设备未向当前环境暴露。

本次检查按题目要求如实记录结果，未使用 `sudo` 安装驱动，也未修改系统级环境。

## 飞书补充文档

- 链接：[雷梓一 - A0 补充材料](https://fudan-nlp.feishu.cn/wiki/AnJwwt2xiiZ9e1kQFibcIivxnsb?from=from_copylink)

该文档设置为组织内公开，用于保存 A0 的最小必要、已脱敏组内验收材料，且未开启互联网公开访问。

## 问题与收获

1. 理解了 `origin` 与 `upstream` 的区别，并掌握了从最新 `upstream/main` 创建独立作业分支的流程。
2. 学会了使用用户级 Python virtual environment 隔离依赖，避免通过 `sudo pip` 修改系统 Python。
3. 通过权限实验理解了 `600` 表示仅文件所有者具有读写权限。
4. 使用 `nohup` 完成了后台进程实验，理解了 SSH 会话与后台任务生命周期之间的关系。
5. 明确了 `gpustat` 安装成功不代表 GPU 查询一定成功；工具仍依赖 NVIDIA 驱动、NVML 和设备访问状态。
6. 认识到命令不存在、未检测到设备和驱动或 NVML 不可用是不同状态，不能在证据不足时直接断言服务器没有 GPU。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] 当前 GitHub 文件不包含任何 Secret、Token、Cookie、密码或私钥。
- [x] 我已将飞书补充文档设置为组织内公开，且没有开启互联网公开访问。