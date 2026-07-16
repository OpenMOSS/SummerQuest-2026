# A0 公开提交：许志鹏

## GitHub 与 PR

- 分支：`a0/Kevin589981`
- Git 操作总结：已完成课程仓库 Fork、本地 clone、添加 `upstream`、同步主分支并创建 `a0/Kevin589981` 分支；`commit`、`push` 和 Pull Request 将在完成文档整理后统一提交。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.2 LTS（x86_64）远程实验环境
- Python：3.12.3
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：`tmux`（另确认环境中可用 `nohup`）


## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- Exit code：0
- 状态类别：成功

### 状态解释

本次环境中 `nvidia-smi` 与 `gpustat` 都成功执行，退出码均为 `0`，说明当前远程环境已正确提供 NVIDIA 驱动、GPU 设备和可访问的查询接口。`nvidia-smi` 直接依赖系统中的 NVIDIA 驱动与设备状态；`gpustat` 则是基于 Python 包调用 NVIDIA 管理库读取同类信息，因此它除了依赖 GPU/驱动环境外，还依赖用户侧 Python 虚拟环境中的相关包安装正确。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/C6ppwgpPxivcWnkQ3ZScXkgenf1

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

本次 A0 主要用于温故实验环境中的基础命令与流程，包括 GitHub 协作、Linux 环境检查、Python virtual environment、文件权限设置、`tmux`/`nohup` 等会话保持方式，以及 `nvidia-smi` 和 `gpustat` 的状态检查与退出码判断。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
