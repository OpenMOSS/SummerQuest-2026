# A0 公开提交：钱张枫

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/MAK1MAAaa`
- Git 操作总结：<说明 fork、upstream、branch、commit、push、PR 的完成情况>

## Linux 环境摘要

- 操作系统：Ubuntu 20.04.6 LTS
- Python：Python 3.8.10
- Virtual environment：已创建
- 模拟密钥文件权限：600 <HOME_DIRECTORY>/<WORK_DIRECTORY>/example.env
- 常驻进程方式：Supervisor

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
NVIDIA-SMI 580.159.03
Driver Version: 580.159.03
CUDA Version: 13.0

GPU 0: NVIDIA A800-SXM4-80GB
GPU 1: NVIDIA A800-SXM4-80GB

已分配设备：CUDA_VISIBLE_DEVICES=0
GPU 显存：81920 MiB
MIG Mode：Disabled
未发现正在运行的 GPU 进程
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
[已脱敏的计算节点] Sun Jul 12 22:21:29 2026  580.159.03
[0] NVIDIA A800-SXM4-80GB | 28°C, 0 % | 0 / 81920 MB |
[1] NVIDIA A800-SXM4-80GB | 20°C, 0 % | 0 / 81920 MB |
```

### 状态解释

`nvidia-smi` 和 `gpustat` 均在 Slurm 分配的 GPU 任务环境中执行成功，Exit code 均为 0。该计算环境能够访问 NVIDIA GPU 设备节点、580.159.03 版 NVIDIA 驱动和 NVML，并识别到 NVIDIA A800-SXM4-80GB GPU。

`nvidia-smi` 是 NVIDIA 驱动工具，主要依赖 `nvidia-smi` 可执行文件、NVIDIA 内核驱动、GPU 设备节点以及 NVML 动态库。登录容器最初没有该命令和完整的 GPU 设备映射，因此无法执行；通过 Slurm 申请 GPU 资源后，计算节点上的驱动环境完整，命令执行成功。

`gpustat` 是 Python 第三方工具，除依赖相同的 NVIDIA 驱动和 NVML 外，还依赖 Python 解释器、`gpustat` 及 `nvidia-ml-py` 等 Python 包。最初通过 `uv tool install` 创建的环境位于登录 Docker 容器中，计算节点无法访问其 Python 解释器，因此返回 Exit code 126。改用 `uvx --isolated` 后，uv 在计算节点可访问的隔离环境中运行 `gpustat` 1.1.1，最终成功获取 GPU 状态并返回 Exit code 0。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/PCaPw2YapiZXI3k6rQ6c67yAn3b?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

### 问题 1

当前环境的 PID 1 是 `supervisord`，不是 `systemd`：

```bash
ps -p 1 -o comm=
```

输出：

```text
supervisord
```

这意味着当前环境可能是容器或受控运行环境，无法正常使用以下用户级 systemd 命令管理常驻进程：

```bash
systemctl --user status
systemctl --user enable --now <服务名>
```

### 解决方法

放弃 `systemd --user` 方案，改用当前环境支持的 Supervisor 管理常驻进程。

由于系统级 Supervisor 已作为 PID 1 运行，不应停止或修改它。应为普通用户创建独立的用户级 Supervisor 实例，使其使用独立的配置文件、控制 socket、PID 文件和日志目录。

### 问题 2

直接执行以下命令：

```bash
supervisorctl status
```

出现权限错误：

```text
PermissionError: [Errno 13] Permission denied
```

原因是未指定配置文件时，`supervisorctl` 会尝试连接系统级 Supervisor，而系统级 Supervisor 的控制 socket 不允许普通用户访问。

不应通过修改系统 socket 权限或停止 PID 1 解决：

```bash
sudo chmod 777 <Supervisor socket>
sudo kill -9 1
```

这些操作可能带来安全风险，甚至导致当前容器或运行环境退出。

### 解决方法

在项目虚拟环境中安装 Supervisor：

```bash
uv pip install supervisor
```

创建独立的用户级 Supervisor 配置，并通过变量保存配置文件位置：

```bash
CONFIG="$HOME/<项目目录>/.supervisor/supervisord.conf"
```

启动用户级 Supervisor：

```bash
.venv/bin/supervisord \
  -c "$CONFIG"
```

查看用户级 Supervisor 管理的进程状态：

```bash
.venv/bin/supervisorctl \
  -c "$CONFIG" \
  status
```

后续所有 `supervisorctl` 命令都应显式指定用户级配置文件：

```bash
.venv/bin/supervisorctl \
  -c "$CONFIG" \
  <子命令>
```

这样可以避免连接系统级 Supervisor，不需要修改系统权限，也不会干扰作为 PID 1 运行的系统级 Supervisor。

### 问题 3

当前 SSH 登录环境直接执行 `nvidia-smi`：

```bash
nvidia-smi
```

输出：

```text
bash: nvidia-smi: command not found
```

进一步检查运行环境：

```bash
cat /etc/os-release
systemd-detect-virt 2>/dev/null || true
lspci | grep -iE 'nvidia|vga|3d'
ls -l /dev/nvidia* 2>/dev/null
```

关键结果：

```text
Ubuntu 20.04.6 LTS
docker
ASPEED Graphics Family
/dev/nvidiactl
```

这意味着当前 SSH 登录 Shell 位于 Docker 或受控登录环境中，而不是已经分配 GPU 的计算任务环境：

- `ASPEED Graphics Family` 是服务器管理显示芯片，不是 CUDA 计算 GPU。
- 只有 `/dev/nvidiactl`，缺少可供任务使用的完整 GPU 设备映射。
- `nvidia-smi` 未注入当前登录环境，因此 Bash 返回 Exit code 127。
- 在该环境中安装 `nvidia-utils` 或 NVIDIA 驱动不能获得 GPU，也不应在无管理员授权时修改系统驱动。

### 解决方法

集群提供 Slurm，因此应先通过调度器申请 GPU，再在分配到的计算任务中执行 GPU 命令：

```bash
command -v sinfo
command -v srun
sinfo -o '%P %G %D %t'
```

使用实际 GPU 分区名运行短时检查任务：

```bash
srun \
  --partition=<GPU分区> \
  --gres=gpu:1 \
  --time=00:05:00 \
  bash -c '
    echo "SLURM_JOB_ID=$SLURM_JOB_ID"
    echo "CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES"

    nvidia-smi
    exit_code=$?

    echo "NVIDIA_SMI_EXIT_CODE=$exit_code"
    exit "$exit_code"
  '
```

脱敏后的关键结果：

```text
CUDA_VISIBLE_DEVICES=0
NVIDIA-SMI 580.159.03
Driver Version: 580.159.03
CUDA Version: 13.0
GPU: NVIDIA A800-SXM4-80GB
NVIDIA_SMI_EXIT_CODE=0
```

最终状态：

```markdown
### `nvidia-smi`

- Exit code：0
- 状态类别：成功
```

`nvidia-smi` 依赖 NVIDIA 内核驱动、GPU 设备节点、NVML 动态库以及对应的命令行程序。Slurm 任务成功提供了这些依赖，因此命令能够识别 GPU 并返回 Exit code 0。

`nvidia-smi` 中显示的 CUDA 版本表示当前驱动支持的最高 CUDA Driver API 版本，不代表当前 Python 环境一定安装了同版本的 CUDA Toolkit。

### 问题 4

在登录环境中使用 uv 安装 `gpustat`：

```bash
uv tool install gpustat
uv tool list
```

安装结果：

```text
gpustat v1.1.1
```

随后在 Slurm 任务中直接运行已安装的命令：

```bash
srun \
  --partition=<GPU分区> \
  --gres=gpu:1 \
  --time=00:05:00 \
  bash -c '
    UV_TOOL_BIN="$(uv tool dir --bin)"
    export PATH="$UV_TOOL_BIN:$PATH"

    gpustat --version
    gpustat
  '
```

出现错误：

```text
<用户目录>/.local/bin/gpustat: <用户目录>/.local/share/uv/tools/gpustat/bin/python: bad interpreter: No such file or directory
GPUSTAT_EXIT_CODE=126
```

根因是 `uv tool install` 在登录 Docker 环境中创建了隔离 Python 环境。`gpustat` 启动脚本虽然能被计算节点看到，但脚本首行引用的 Python 解释器只存在于登录环境，计算节点无法访问该解释器。

Exit code 126 表示 Shell 找到了命令，但命令无法执行。它不等同于以下状态：

- Exit code 127：命令不存在。
- 未检测到 GPU。
- NVML 或 NVIDIA 驱动不可用。

### 解决方法

使用 `uvx --isolated` 在 Slurm 计算任务可访问的隔离环境中运行固定版本的 `gpustat`：

```bash
srun \
  --partition=<GPU分区> \
  --gres=gpu:1 \
  --time=00:05:00 \
  bash -c '
    uvx --isolated --from "gpustat==1.1.1" gpustat --version

    uvx --isolated --from "gpustat==1.1.1" gpustat
    exit_code=$?

    echo "GPUSTAT_EXIT_CODE=$exit_code"
    exit "$exit_code"
  '
```

脱敏后的关键结果：

```text
gpustat 1.1.1
[已脱敏的计算节点] 580.159.03
[0] NVIDIA A800-SXM4-80GB | 28°C, 0 % | 0 / 81920 MB |
[1] NVIDIA A800-SXM4-80GB | 20°C, 0 % | 0 / 81920 MB |
GPUSTAT_EXIT_CODE=0
```

最终状态：

```markdown
### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功
```

`gpustat` 除依赖 NVIDIA 驱动、GPU 设备节点和 NVML 外，还依赖可执行的 Python 环境、`gpustat` 包及 `nvidia-ml-py` 等 Python 依赖。`uvx --isolated` 在计算节点可访问的环境中解析并运行这些依赖，因此最终返回 Exit code 0。

即使状态工具能够列出多张物理 GPU，任务也只能使用 Slurm 实际分配的资源。计算程序应遵守 `CUDA_VISIBLE_DEVICES`，不得主动访问未分配的 GPU。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
