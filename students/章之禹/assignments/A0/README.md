# A0 公开提交：章之禹

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/zhiyuzhang-0212`
- Git 操作总结：均已完成

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：Python 3.12.11
- Virtual environment：已创建
- 模拟密钥文件权限：应为 600
- 常驻进程方式：tmux,nohup

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
<可选：粘贴已删除主机名、用户名、进程、内部路径等信息的关键输出>
```

### `gpustat`

- 安装版本：gpustat 1.1.1
- Exit code：0
- 状态类别：成功

```text
<可选：粘贴已脱敏的关键输出>
```

### 状态解释

`nvidia-smi` 能够成功执行，说明系统中安装了 NVIDIA 驱动，并且驱动能够通过 NVML（NVIDIA Management Library）正常获取 GPU 的硬件信息。`gpustat` 也能够正常执行，因为它是一个 Python 工具，底层同样调用 NVML 获取 GPU 的状态信息，因此依赖正确安装的 NVIDIA 驱动和可用的 NVML 库。

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

<总结 2-5 个实际遇到的问题、排查方法和学到的内容。只写可公开信息。>

## 自检

- [√] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [√] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [√] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [√] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [√] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
