# A0 公开提交：曹正伟

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/OpenCeadar`
- Git 操作总结：fork、upstream、branch、commit、push、PR 全部完成

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：Python 3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux (version 3.2a)

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
bash: nvidia-smi: command not found
```

### `gpustat`

- 安装版本：gpustat 0.6.0
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag for details
```

### 状态解释

- `nvidia-smi` 失败是因为当前服务器环境中没有这个命令，说明系统没有安装或没有暴露 NVIDIA 的系统管理工具。`nvidia-smi` 依赖系统级 NVIDIA 驱动栈以及对应的命令行工具。如果当前节点是 CPU 服务器，出现这个结果是合理的，因此我没有尝试安装系统级 GPU 驱动或修改系统环境。

- `gpustat` 是在我的用户级 Python virtual environment 中安装并成功运行的，但它在查询 NVIDIA 设备时失败了。这说明 `gpustat` 命令本身可用，问题不在 Python 包是否安装成功，而在底层 GPU 查询能力不可用。`gpustat` 依赖 Python 包本身以及底层的 NVIDIA Management Library (NVML)、NVIDIA 驱动和可访问的 NVIDIA 设备。当前环境中可能没有可访问的 GPU，或没有可用的 NVML/驱动支持。

## 飞书补充文档

- 链接：(https://fudan-nlp.feishu.cn/wiki/G4PxwIb3wib0GHkPVADc8zQSnPe?from=from_copylink)

已经将文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 熟悉了ssh远程运行的操作
- 熟悉了vscode的运行操作
- 学习了配置VPN及其使用

## 自检

- [ ] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [ ] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [ ] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [ ] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [ ] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
