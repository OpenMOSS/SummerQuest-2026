# A0 公开提交：张俊鹏

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/lalalalulu2`
- Git 操作总结：已全部完成

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.5 LTS
- Python：Python 3.12.13
- Virtual environment：已创建
- 模拟密钥文件权限：-rw------- 1 root root 0 Jul 11 07:27 mock_secret.env
- 常驻进程方式：学了 tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
<可选：粘贴已删除主机名、用户名、进程、内部路径等信息的关键输出>
```

### `gpustat`

- 安装版本：26.1.2
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

我启动的是 CPU 资源空间的实例。nvidia-smi 能够成功，代表了查询成功，但是没有任何输出，状态是成功；gpustat 由于当前环境没有可用 gpu，退出码为 1，并且输出了相应报错。

## 飞书补充文档

- 链接：<粘贴飞书 Doc 或 Wiki 链接>

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. 之前经常使用GitHub，但是对于很多 git 的操作不熟悉，以及对于团队协作的流程更加熟悉了
2. 启智上的 /root 目录是每个容器特有的

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
