# A0 公开提交：胡宸

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/HuChen03`
- Git 操作总结：已在个人 fork 中创建 A0 分支，并配置 `origin` 与 `upstream`；本次修改范围限制在 `students/胡宸/` 下，后续按要求 commit、push 并创建 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04 LTS，Linux x86_64
- Python：3.12.13
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
nvidia-smi: command not found
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

`nvidia-smi` 是 NVIDIA 驱动工具的一部分，当前环境找不到该命令，因此无法通过它查询 GPU。`gpustat` 已安装在用户级 virtual environment 中，但它依赖 NVML 读取 NVIDIA 设备状态；当前环境缺少 NVML 共享库，所以查询失败。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/XoUYwJIQJiZrGrkKV5ucJROHnFe?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 模拟敏感配置文件权限应设为 `600`，只允许当前用户读写。
- 公开报告需要脱敏，不提交用户名、主机名、IP、内部路径、进程参数或凭据页面。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
