# A0 公开提交：于康

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/kaysonyu`
- Git 操作总结：已完成课程仓库 Fork，将个人 Fork clone 到工作区，并分别配置个人 Fork 为 `origin`、OpenMOSS 课程仓库为 `upstream`。
- 分支基线：当前作业分支从与最新 `upstream/main` 对齐的主分支创建。
- 后续提交约定：文档确认和检查完成后，使用 Conventional Commits 提交并 push 到个人 Fork，再向上游 `main` 创建 PR；PR 标题为 `[A0] 于康 - 完成基础环境与 Profile`。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.2 LTS（x86_64）
- Python：3.12.3
- Virtual environment：已创建
- Python 依赖管理：`gpustat 1.1.1` 安装在上述 virtual environment 中；未使用 `sudo pip`，也未修改系统 Python。
- 模拟密钥文件权限：`600`
- 常驻进程方式：`tmux 3.4`；已核验 detached session 可以独立于当前终端保留。

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：`0`
- 状态类别：命令入口异常，未发生有效的 NVIDIA 设备查询。

```text
（无标准输出或标准错误）
```

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 共享库不可用，当前无法查询 NVIDIA 设备。

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 通常是 NVIDIA 驱动工具链的一部分，查询过程依赖有效的可执行程序、NVIDIA 驱动、设备节点和 NVML。本次 Shell 返回退出码 `0`，但该结果只表示空的命令入口执行结束，不表示 NVIDIA 工具已完成 GPU 查询，也不能用来证明驱动或 GPU 可用。

`gpustat` 命令本身已经能在用户级 virtual environment 中启动，但它还需要通过 `nvidia-ml-py` 访问底层 NVML。本次退出码为 `1`，并明确报告 `NVML Shared Library Not Found`，说明当前 NVML 查询链路不可用。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/FvixdEyqioZXzyxBZiNcMXU9nMd

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. 命令退出码必须和输出、命令入口以及依赖状态结合判断。本次 `nvidia-smi` 的反例说明，退出码 `0` 不必然表示工具完成了有效查询。
2. `gpustat` 安装成功不等于 GPU 查询一定成功；它仍依赖 NVIDIA 驱动、NVML 共享库和设备访问条件。
3. Python 依赖应安装在用户级 virtual environment 中，避免污染系统 Python，也不应使用 `sudo pip`。
4. 权限 `600` 表示只有文件所有者可读写，适合保护本地敏感配置；示例文件应保留在用户工作区，不应提交到 Git。
5. `tmux` 能把会话与当前 SSH 终端解耦，但这与系统重启后自动恢复进程是两个不同问题。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
