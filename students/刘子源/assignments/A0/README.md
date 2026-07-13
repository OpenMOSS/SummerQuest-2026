# A0 公开提交：刘子源

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/Zyrrick`
- 已完成课程仓库 Fork，并在个人服务器中 clone 仓库。
- 已将实验室课程仓库配置为 `upstream`，并基于最新主分支创建个人作业分支。
- 本次修改仅位于个人学生目录；已使用 Conventional Commits 完成 commit、push，并创建 Pull Request。

## Linux 环境摘要

- 操作系统：Linux 5.15.0-119-generic x86_64
- Python：3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：已了解 tmux：可创建持久会话，SSH 断开后进程仍可继续运行，并可重新 attach。

## 查看cpu、内存和磁盘

- CPU、内存与个人 home 目录磁盘状态：已完成检查，详细脱敏记录见飞书 A0 补充文档。

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

`nvidia-smi` 的退出码为 127，且 Shell 提示 `command not found`。这表示当前环境无法找到该命令，可能是 NVIDIA 驱动配套工具未安装，或该命令不在当前环境的 `PATH` 中；仅凭这一结果不能直接判断服务器一定没有 NVIDIA GPU。
`gpustat` 已在用户级 virtual environment 中安装成功，但执行时返回退出码 1，并提示 `Driver Not Loaded`。`gpustat` 需要通过 NVML 查询 NVIDIA GPU 状态；该结果表示当前环境没有可用的 NVIDIA 驱动或 NVML 查询能力，因此无法读取 GPU、显存和相关进程信息。
综上所述，当前个人 CPU 服务器不具备可用的 NVIDIA GPU 查询环境。

## 飞书补充文档

- 飞书个人主页：已创建并设置为组织内公开；主页链接及权限状态已登记在公开 `PROFILE.md` 中。
- 飞书 A0 补充文档：已创建并设置为组织内公开；未开启互联网公开访问。

[A0 飞书补充文档](https://fudan-nlp.feishu.cn/wiki/KTjpwGhVQiuj3wkzZBocAihan4c?from=from_copylink)

## 问题与收获

- `tmux` 可用于维持持久会话，使 SSH 断开后会话中的任务继续运行，并可在重新连接后恢复会话。
- 公开材料中不应包含服务器账号、主机名、内部路径、代理地址或任何凭据。

## 自检

- [ 完成 ] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [ 完成 ] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [ 完成 ] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [ 完成 ] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [ 完成 ] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
