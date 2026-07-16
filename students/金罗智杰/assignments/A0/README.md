# A0 公开提交：金罗智杰

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/jinluo12345`
- Git 操作总结：已 Fork 课程仓库并 clone 个人 Fork，已将课程官方仓库配置为 `upstream`，并从最新 `upstream/main` 创建 `a0/jinluo12345` 分支。作业内容使用 Conventional Commits 提交，分支推送至个人 Fork，并通过 Pull Request 提交到课程仓库。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04 LTS（x86_64）
- Python：3.12.12
- Virtual environment：已在用户级环境中创建
- 模拟密钥文件权限：`600`
- 常驻进程方式：`nohup`

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
命令已实际执行成功；设备型号和状态明细未放入公开报告。
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
命令已实际执行成功；GPU 型号、显存、温度、利用率和进程明细未放入公开报告。
```

### 状态解释

`nvidia-smi` 依赖 NVIDIA 驱动与 NVML；`gpustat` 通过 Python 包及 NVML 接口读取 GPU 状态。两个命令均返回 0，说明本次执行时命令可用、驱动可访问且成功检测到设备。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/Mh8nwFlGMi4iHgkm6bMcRLIDn6c

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. 理解了 `nvidia-smi` 和 `gpustat` 对驱动、NVML 及 Python 环境的不同依赖，并在公开报告中主动移除硬件和运行状态明细。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
