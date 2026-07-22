# A0 公开提交：锁祎然

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/YiRaaaan`
- 已完成课程仓库 Fork，clone 到个人服务器，并将实验室仓库配置为 `upstream`。
- 基于最新 `upstream/main` 创建个人作业分支 `a0/YiRaaaan`，本次修改仅位于 `students/锁祎然/` 目录。
- 使用 Conventional Commits 完成 commit、push，并提交 Pull Request。

## Linux 环境摘要

- 操作系统：Linux 5.15.0-78-generic x86_64（Ubuntu 22.04 系列内核）
- Python：3.10.12
- Virtual environment：已创建（用户级 virtualenv；系统 `python3.10-venv` 未预装且不使用 sudo，改为 `pip install --user virtualenv` 后创建 venv）
- 模拟密钥文件权限：600
- 常驻进程方式：tmux 3.2a（`tmux new -s work` → detach，SSH 断开后进程持续运行；重登后 `tmux attach -t work` 恢复）

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
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 退出码 127 且 shell 提示 `command not found`，说明当前用户环境的 `PATH` 中不存在该命令。`nvidia-smi` 是 NVIDIA 驱动配套的用户态工具，安装 GPU 驱动时才会一并存在；仅凭这一结果无法断言硬件层面一定没有 GPU，但可以断言当前用户环境不具备 NVIDIA 用户态工具。

`gpustat` 通过 NVML（`libnvidia-ml.so`）查询 NVIDIA GPU；已在用户级 venv 中安装成功但执行退出码 1，报 `NVML Shared Library Not Found`，说明系统没有安装 NVIDIA 驱动，NVML 共享库不存在。综合两条结果可以确认当前个人服务器是一台 CPU 机器，无 NVIDIA 用户态工具也无 NVML，符合 A0 说明中"个人 CPU 服务器可能没有 NVIDIA GPU"的场景；未为了让命令成功而安装系统级驱动。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/Qf8fwKmp1iCxheknuu5clmuinqb

该文档设置为组织内公开，用于保存 A0 的组内验收材料（含更完整的环境记录、gpustat 与 nvidia-smi 原始输出、常驻进程验证、以及排查过程）。

## 问题与收获

- **系统 `python3-venv` 缺失**：直接 `python3 -m venv .venv` 报 `ensurepip is not available`。规则禁用 `sudo pip` / `sudo apt`，改用 `pip install --user virtualenv` + `virtualenv .venv` 创建用户级 venv，不触碰系统包，仍然满足"在自己的 home 目录管理 Python 环境"的目标。
- **gpustat 与 nvidia-smi 的两种失败模式**：一个是"命令根本不存在"（exit 127），一个是"命令能跑但 NVML 缺失"（exit 1）。这一组合恰好构成 CPU 机上"用户态工具 + 内核态驱动"两层缺失的完整证据链，比只依赖 `command not found` 更能说明问题。
- **不为通过命令而修改系统环境**：A0 明确说明 CPU 机器上跑不通不影响得分，关键是如实执行、记录、解释。避免了通过安装假驱动或修改 PATH 来"制造"成功输出的诱惑。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
