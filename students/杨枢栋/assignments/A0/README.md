# A0 公开提交：杨枢栋

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/luppppy`
- Git 操作总结：fork、upstream、branch、commit、push、PR已完成

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：Python 3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- 安装版本：gpustat 1.1.1
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 是 NVIDIA 驱动提供的命令行工具，用于读取 GPU、显存、驱动版本和进程状态。`gpustat` 是用户级 Python 工具，也依赖 NVML 来查询 NVIDIA 设备状态。
本次检查中，`nvidia-smi` 可以正常显示 GPU 状态，说明系统层面的 NVIDIA 驱动和设备可用；但`gpustat` 在当前 Python 环境中查询设备时提示 `NVML Shared Library Not Found`，退出码为 `1`。这说明 Python 工具运行环境没有正确找到 NVML 共享库，可能与动态库路径有关。根据 A0 要求，我只记录命令输出、退出码和判断结果，没有使用 `sudo` 安装驱动或修改系统级 GPU 环境。

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

本次作业让我熟悉了 Fork、upstream、分支、PR 和用户级环境管理的基本流程，也提醒我公开提交前需要主动删除用户名、主机名、内部路径、进程参数和凭据信息。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
