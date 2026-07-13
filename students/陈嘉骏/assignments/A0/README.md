# A0 公开提交：陈嘉骏

## GitHub 与 PR

- 分支：`a0/ljbro`
- Git 操作总结：fork、upstream、branch、commit、push、PR已完成

## Linux 环境摘要

- 操作系统：Linux 5.15.0-119-generic x86_64
- Python：3.12.2
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
```

### 状态解释

nvidia-smi失败是因为没有安装NVIDIA驱动，它依赖NVIDIA的显卡驱动套件
gpustat失败也是因为没有安装NVIDIA驱动，它本身不能读取硬件，也需要靠驱动获取数据。除了驱动，它还依赖python环境、库

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/PZ4JwF0N2iQapKk2AtPc1fuWn2d?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

知道了几个之前没用过的linux命令

## 自检

- [√] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [√] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [√] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [√] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [√] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
