# A0 公开提交：朱凡

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/Bamboovan`
- Git 操作总结：已 fork 仓库到 `Bamboovan/SummerQuest-2026`（origin），添加 upstream 远程指向 `OpenMOSS/SummerQuest-2026`，从 main 创建分支 `a0/Bamboovan`，完成 commit 并 push 到 origin，向上游发起 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：3.10.12
- Virtual environment：已创建（conda）
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

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
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
Driver Not Loaded
```

### 状态解释

`nvidia-smi` 是随 NVIDIA 驱动附带安装到 `/usr/bin/nvidia-smi` 的系统命令，本身是驱动用户态工具的一部分。当前服务器没有安装 NVIDIA 驱动，所以该二进制文件不存在，bash 返回退出码 127（标准"命令未找到"退出码）。

`gpustat` 是一个 Python 包（已通过 pip 在 cs336 环境中安装成功，版本 1.1.1），它通过 NVML（NVIDIA Management Library）的 Python 绑定查询 GPU 状态。命令本身能运行，但运行时需要通过 NVML 访问已加载的 NVIDIA 内核驱动；当前系统未加载 `nvidia` 内核模块（"Driver Not Loaded"），因此 NVML 无法初始化，退出码 1。

两者的依赖链不同：`nvidia-smi` 依赖驱动附带的二进制文件是否存在，驱动没装则命令直接缺失；`gpustat` 依赖 Python + NVML 库 + 已加载的内核驱动，命令本身可以装上，但运行时仍需驱动在场。当前机器是 Slurm 集群的控制节点，通常只负责作业调度，不配备 GPU，因此两个命令都失败属于预期情况，要使用 GPU 需通过 `srun`/`salloc` 申请计算节点。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/K4epwQTdcikGx5kpwracaCagnHf

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. **conda 创建环境时报 ToS 未接受错误**：新版 conda 要求先接受 Anaconda 默认频道的服务条款才能创建环境。排查后改用 `-c conda-forge --override-channels` 指定 conda-forge 频道，既绕过 ToS 限制，也避免 Anaconda 商业频道的许可问题，更适合长期使用。

2. **GPU 检查命令全部失败**：`nvidia-smi` 返回 127（命令不存在），`gpustat` 返回 1（Driver Not Loaded）。最初怀疑环境配置问题，确认所在机器是 Slurm 集群的控制节点（slurmctld），只负责作业调度、不配备 GPU，要使用 GPU 需通过 `srun`/`salloc` 申请计算节点。学到 Slurm 集群的节点分工，以及 `nvidia-smi`（随驱动附带的系统二进制）和 `gpustat`（Python 包，运行时通过 NVML 访问驱动）依赖链的差异--前者驱动没装则命令缺失，后者能装上但运行时仍需驱动在场。

3. **conda base 环境遮蔽系统 Python**：`python3 --version` 显示 3.14.6 而非 Ubuntu 22.04 默认的 3.10，原因是 conda 默认 `auto_activate_base=yes`，激活后 base 的 Python 会遮蔽系统 Python。通过 `which python3` 和 `conda deactivate` 验证了两者指向不同二进制。学到 conda 环境激活对 PATH 的修改机制，以及如何用 `conda config --set auto_activate_base false` 关闭自动激活。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
