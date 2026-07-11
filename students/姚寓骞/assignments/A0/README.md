# A0 公开提交：姚寓骞

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/nightwatcher-of-abyss`
- Git 操作总结：已完成课程仓库 Fork，并配置 `upstream`，从最新 `upstream/main` 创建独立 A0 分支，并仅修改自己的学生目录；使用 Conventional Commit 提交并通过 Pull Request 交付。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.1 LTS
- Python：Python 3.12.3
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux/nohup

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功
- 公开输出：不提交具体设备名称或状态明细

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功
- 公开输出：[0] NVIDIA H100 80GB HBM3 | 32°C,   0 % |     0 / 81559 MB |

### 状态解释

`nvidia-smi` 能够成功执行，说明当前环境中的 NVIDIA 驱动与 NVML 可以正常返回GPU状态。`gpustat` 是 Python 层工具，同样依赖 NVML，成功执行说明安装的包和底层查询链路均可用。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/QzjYdsobhohVxFxwNfUc9ahRnhe

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 通过 Fork、upstream、独立分支和 Pull Request 走通标准规范的 GitHub 协作流程，比之前参与的小规模开发更严谨。
- 了解了 `nvidia-smi`、`gpustat` 与 NVML 的依赖关系


## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
