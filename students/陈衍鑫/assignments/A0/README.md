# A0 公开提交：陈衍鑫

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/SSSSuperC`
- Git 操作总结：已从公开仓库创建本地工作副本，将个人 fork 配置为 `origin`、OpenMOSS 主仓配置为 `upstream`，使用脚手架生成 `students/陈衍鑫/`，并在 `a0/SSSSuperC` 分支整理 A0 提交内容。提交范围只包含个人目录，已 push 到个人 fork；PR 从该分支创建并指向上游 `main`。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS，Linux x86_64
- Python：3.12.2
- Virtual environment：已创建；本次在用户可写临时目录中完成，未使用 `sudo pip`
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：命令可执行，但当前环境未返回可用 GPU 信息

```text
命令无标准输出。
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML 或驱动不可用

```text
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 是 NVIDIA 驱动侧的状态查询工具；本环境中命令存在并返回 0，但没有输出 GPU 列表或状态信息，因此不能据此判断当前环境有可用 GPU。

`gpustat` 通过 Python 包和 NVML 查询 NVIDIA 设备状态。本次 `gpustat` 能正常安装和启动，但查询阶段无法加载可用的 NVML，因此返回错误。A0 不要求 GPU 检查一定成功；这里记录的是当前执行环境下真实运行两个命令后的结果。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/Te1ldC6bEoAp1Nx11fLcI4b2nNh

该文档用于保存 A0 的组内验收材料；权限已设置为组织内获得链接的人可阅读，且未开启互联网公开访问。

## 问题与收获

- 当前 Codex 沙箱不允许写入 home 目录，因此本次在用户可写临时目录完成 A0 的目录、virtual environment 和模拟密钥权限操作。
- 安装 `gpustat` 时需要访问 Python 包源；在网络权限放通后安装成功。
- `nvidia-smi` 与 `gpustat` 的表现不完全一致：前者命令可执行但没有输出，后者明确提示 NVML 不可用。后续判断 GPU 状态时需要区分“命令存在”和“能否通过 NVML 查询设备”。
- 整理公开报告时，需要主动删除用户名、主机名、IP、内部路径、完整进程参数和硬件容量等不适合公开的信息。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
