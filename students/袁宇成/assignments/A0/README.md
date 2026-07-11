# A0 公开提交：袁宇成

> 本文件公开可见，只记录脱敏结果。具体显卡名称、型号、数量、UUID、利用率和进程信息均不进入 commit、PR 或报告。

## GitHub 与 PR

- 分支：`a0/southwindyong`
- Git 操作总结：已完成课程仓库 Fork，配置 `upstream`，从最新 `upstream/main` 创建独立 A0 分支，并仅修改自己的学生目录；使用 Conventional Commit 提交并通过 Pull Request 交付。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.2 LTS
- Python：Python 3.12.3
- Virtual environment：已创建并验证
- 模拟密钥文件权限：600（仅当前用户可读写）
- 常驻进程方式：了解并选择 `nohup`，实际使用时配合日志与 PID 管理

公开材料不包含用户名、主机名、IP、内部路径、SSH 配置、硬件容量或进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功
- 公开输出：不提交具体设备名称或状态明细

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功
- 公开输出：不提交具体设备名称或状态明细

### 状态解释

`nvidia-smi` 能够成功执行，说明当前环境中的 NVIDIA 驱动与 NVML 可以正常返回设备状态。`gpustat` 是 Python 层工具，同样依赖 NVML；其成功执行说明 Python 包和底层查询链路均可用。两项检查只记录退出码和状态类别，不记录具体显卡名称、型号、数量、UUID、利用率或进程区。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/YA2idW7p3oBkJvxZojXcp9KQnFr
- 权限状态：组织内获得链接的人可阅读

该文档保存 A0 组内验收所需的最小脱敏记录，不保存凭据和具体显卡信息。

## 问题与收获

- 首次安装依赖时，系统预配置的 Python 包镜像要求认证。确认 virtual environment 正常后，仅对隔离环境的一次安装命令临时指定公开 PyPI，未索取或保存镜像凭据。
- 理解了 `nvidia-smi`、`gpustat` 与 NVML 的依赖关系，并能依据退出码区分命令成功、命令不存在或驱动不可用等状态。
- 完成了用户级 virtual environment、敏感配置权限和当前用户进程检查，进一步明确了公开、组内与机密材料的边界。
- 通过 Fork、upstream、独立分支和 Pull Request 走通标准 GitHub 协作流程。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、硬件明细、进程参数和组内数据。
- [x] commit、PR、GitHub 与飞书正文均未记录具体显卡名称、型号、数量、UUID、利用率或进程信息。
- [x] GitHub 和飞书正文都没有 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档为组织内链接可读，未开启互联网公开链接。
