# A0 公开提交：马万里

## GitHub 与 PR

- 分支：`a0/maerciyuan`
- 已 fork 课程仓库并将个人 fork 配置为 origin，将课程仓库配置为 upstream；从 upstream/main 最新状态创建 `a0/maerciyuan` 分支，使用 Conventional Commits 规范提交作业后 push 到个人 fork，并在 GitHub 网页端向上游主分支发起 Pull Request。

## Linux 环境摘要

- 操作系统：``Ubuntu 22.04.4 LTS``
- Python：Python 3.12.4
- Virtual environment：通过`conda create -n a0`创建完成
- 模拟密钥文件权限：将文件的权限设置为600
- 常驻进程方式：nohup

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：NVIDIA驱动未安装

```text
bash: nvidia-smi: command not found
```

### `gpustat`

- 安装版本：`gpustat 1.1.1`
- Exit code：1
- 状态类别：导入 pynvml 失败或无设备

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
Driver Not Loaded
```

### 状态解释

```
nvidia-smi
```

**原因**：执行失败，系统未安装 NVIDIA 驱动，因此 `nvidia-smi` 这个二进制工具根本没有被安装到系统中，shell 无法找到并执行该命令。

**依赖**：`nvidia-smi` 依赖 NVIDIA 官方驱动，驱动安装时会同时安装 `nvidia-smi` 工具和 NVML 库。

```
gpustat
```

**原因**：`gpustat` 作为 Python 包虽然已经安装，但其底层依赖的 NVML 库不可用——因为 NVIDIA 驱动未安装，所以 NVML 库不存在或无法初始化。`gpustat` 在启动时尝试调用 NVML 检测 GPU 设备，检测失败后以退出码 1 终止。

**依赖**：`gpustat` 依赖 Python 环境 + pynvml 包（NVIDIA 官方 Python 绑定）+ 系统级的 NVML 库（由 NVIDIA 驱动提供）

## 飞书补充文档

https://fudan-nlp.feishu.cn/wiki/KEiFwmx0biCzaMky52mcrQThnVh?from=from_copylink

## 问题与收获

学习到了基础 GPU 环境配置与校验。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。

- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。

- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。

- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。

- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。