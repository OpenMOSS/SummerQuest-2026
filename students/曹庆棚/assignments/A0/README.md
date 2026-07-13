# A0 公开提交：曹庆棚

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/zero0307`
- Git 操作总结：已将课程仓库 Fork 到个人GitHub账号，并在本地配置个人Fork为'origin'、课程原仓库为'upstream'。从最新的'upstream/main'创建了独立作业分支a0/zero0307，仅在个人学生目录中完成A0内容。提交信息使用'Conventional Commits'格式，随后将分支push到个人Fork，并创建了符合命名规范的'A0 Pull Request'。

## Linux 环境摘要

- 操作系统：Ubuntu 20.04.6 LTS, Linux x86_64
- Python：3.8.10
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
bash: nvidia-smi: command not found
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

nvidia-smi执行失败并返回退出码127，说明当前服务器环境中不存在该命令。nvidia-smi是由NVIDIA驱动工具提供的系统级命令，需要服务器安装NVIDIA驱动并具备可访问的NVIDIA GPU。
gpustat已成功安装在Python virtual environment中，但执行时返回退出码1，并提示 NVML Shared Library Not Found。这说明 gpustat程序本身可以启动，但无法找到用于读取GPU状态的 NVML 动态库。gpustat虽然是Python包，但底层仍依赖NVIDIA驱动提供的 NVML，以及当前环境中可见的NVIDIA GPU。
综合来看，当前个人CPU服务器未安装或未提供NVIDIA驱动、NVML 和可访问的NVIDIA GPU，因此两个命令均无法正常查询GPU状态。该结果符合CPU服务器的环境特点。我仅记录并分析了实际结果，没有使用sudo安装驱动，也没有修改系统级环境。


## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/Slq4wXLqviwXHRkGUAjcgRc3naf?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 创建 Python 虚拟环境时遇到 ensurepip 缺失，改用用户级 virtualenv 成功完成环境隔离，未使用 sudo 修改系统环境。
- nvidia-smi 命令不存在，gpustat 因 NVML 不可用而执行失败，确认当前 CPU 服务器没有可访问的 NVIDIA GPU 环境。
- 学会了在目标命令执行后立即使用 $? 记录退出码，避免被后续命令覆盖。
- 熟悉了 virtual environment、文件权限 600、tmux 会话管理以及公开材料脱敏等基础操作。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
