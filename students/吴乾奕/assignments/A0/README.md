# A0 公开提交：吴乾奕

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/WQY5100`
- Git 操作总结：已完成个人 Fork，将个人 Fork 配置为 `origin`、课程仓库配置为 `upstream`，并从最新 `upstream/main` 创建 `a0/WQY5100` 分支；本次作业仅修改 `students/吴乾奕/`，使用 Conventional Commit 提交并推送到个人 Fork，随后由本人向课程仓库 `main` 创建 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04
- Python：3.13.11
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
命令成功。未公开设备、利用率和进程详情。
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 正常返回，说明当前环境可以调用系统提供的 NVIDIA 状态查询工具。`gpustat` 已成功安装，但其 Python NVML 绑定无法加载所需的 NVML 共享库，因此返回退出码 1。该结果表示 `gpustat` 的查询链路当前不可用，不等同于服务器未检测到 GPU。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/Hl7gwOgU9ibGe2k9BgdcJaFmneh

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 通过 `origin` 与 `upstream` 的分工，明确了个人 Fork 和课程仓库的同步、提交与 PR 流程。
- 使用用户级 virtual environment 隔离 Python 依赖，避免修改系统 Python 环境。
- 将模拟密钥文件权限设置为 `600`，理解了仅文件所有者可读写的权限含义。
- 对比 `nvidia-smi` 与 `gpustat` 的结果后，确认不同 GPU 检查工具可能因依赖链路不同而呈现不同状态。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
