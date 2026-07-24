# A0 公开提交：谌奕同

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/bondtesty`
- Git 操作总结：
  1. Fork `OpenMOSS/SummerQuest-2026` 到个人 GitHub 账号 `bondtesty`。
  2. 添加 upstream：`git remote add upstream https://github.com/OpenMOSS/SummerQuest-2026.git`。
  3. 从最新 `main` 创建分支 `a0/bondtesty`。
  4. 使用 `scripts/create_student.py --name 谌奕同 --github bondtesty` 创建学生目录。
  5. 将学生目录和 A0 README 提交并 push 到 fork。
  6. 向 upstream 发起 PR：`[A0] 谌奕同 - 完成基础环境与 Profile`。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04 LTS（容器环境）
- Python：3.10.12
- Virtual environment：已创建 `~/a0-venv`
- 模拟密钥文件权限：`~/.fake_secret` 已设为 600
- 常驻进程方式：`tmux` 可用；`nohup` 也可用

未填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

```text
nvidia-smi: command not found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：`1`
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

这台 notebook 是 CPU-only 实例，没有 NVIDIA GPU，也没有安装 NVIDIA 驱动或 NVML 库。`nvidia-smi` 的退出码 127 表示命令不存在；`gpustat` 虽然安装成功，但它依赖 NVML 查询 GPU 状态，因找不到 NVML 共享库而退出码 1。

这与 A0 题面预期一致：个人 CPU 服务器可能没有 NVIDIA GPU，关键是如实执行、记录退出码并判断原因，而不是为了让命令成功去安装系统级驱动或修改系统环境。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/GEGodTo6go6OTOxHD26cwhYSnhh

该文档设置为组织内公开，用于保存 A0 的组内验收材料，包括更详细的 Linux 环境检查结果、GPU 命令输出与排查记录。

## 问题与收获

1. **inspire notebook 创建后 FAILED**：最初使用 `-r 4CPU` 创建时 notebook 进入 `FAILED`。经与 inspire-cli 作者排查，发现是当时平台调度的短暂异常，后续按 `--workspace cpu --compute-group CPU资源` 创建成功。
2. **inspire notebook ssh 连接失败**：首次 SSH 时 `INSPIRE_SETUP_SCRIPT` 环境变量指向了其他用户的路径，导致 dropbear bootstrap 失败。取消该环境变量后 SSH 成功。
3. **镜像缺少 python3-venv 和 pip**：CPU 基础镜像未预装 `python3-venv`、`python3-pip` 和 `tmux`，通过包管理器安装后完成 venv 创建和 gpustat 安装。
4. **CPU 服务器无 GPU**：`nvidia-smi` 和 `gpustat` 均因无 NVIDIA 驱动/GPU 而失败，符合预期，未使用 sudo 安装驱动或修改系统环境。
5. **体会到双层提交的意义**：公开 README 只保留脱敏摘要，详细命令输出和内部路径放在组织内公开的飞书文档中，避免敏感信息进入 GitHub。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
