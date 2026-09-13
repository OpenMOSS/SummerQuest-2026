# A0 公开提交：何佳莹

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/y-tarl`
- Git 操作总结：已完成课程仓库 Fork，将个人 Fork 配置为 `origin`，OpenMOSS 课程仓库配置为 `upstream`，并基于最新主分支创建本次作业分支。已使用 Conventional Commits 提交并 push 到个人 Fork，再向上游 `main` 创建 Pull Request。

## Linux 环境摘要

- 操作系统：Linux 5.15.0-119-generic x86_64
- Python：3.12.3
- Virtual environment：已创建
- Python 依赖管理：`gpustat 1.1.1` 安装在用户级 virtual environment 中，未使用 `sudo pip`
- 模拟密钥文件权限：600
- 常驻进程方式：`tmux` 可用

## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

```text
command not found
```

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

本次环境中无法找到 `nvidia-smi` 命令，因此返回退出码 127；这只能说明当前命令入口不可用，不能据此判断物理机一定没有 NVIDIA GPU。`gpustat` 已经能够启动，但需要通过 NVML 访问 NVIDIA 驱动和设备，当前缺少可用的 NVML 共享库，所以返回退出码 1。综合来看，当前环境没有可用的 NVIDIA GPU 查询链路。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/Grkkd1bOkowEbvxg20RctH18nmf

该文档设置为组织内公开，用于保存 A0 的组内验收材料，未开启互联网公开访问。

## 问题与收获

1. 命令退出码需要结合标准输出和依赖状态理解，退出码本身不能代替完整诊断。
2. `gpustat` 安装成功不等于 GPU 查询成功；它仍依赖 NVIDIA 驱动、NVML 共享库和设备访问条件。
3. Python 依赖安装在用户级 virtual environment 中，可以避免修改系统 Python，也不需要使用 `sudo pip`。
4. 公开材料需要删除用户名、主机名、IP、内部路径和凭据，只保留能够复核作业完成情况的信息。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
