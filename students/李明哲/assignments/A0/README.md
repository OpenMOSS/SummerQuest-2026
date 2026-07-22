# A0 公开提交：李明哲

## GitHub 与 PR

- 分支：`a0/Mubuky`
- 已将个人 Fork 配置为 `origin`，课程仓库配置为 `upstream`，并基于 `upstream/main` 创建独立分支。
- A0 变更已使用 Conventional Commit 提交并推送到个人 Fork。

## Linux 与 Python 检查

- 操作系统与架构：Ubuntu 22.04，x86_64
- Python：3.12.13
- Virtual environment：已在临时工作区创建并使用，检查结束后清理
- 模拟敏感文件：空测试文件权限为 mode `600`，未写入真实 Secret
- 用户进程：已执行当前用户进程检查，公开记录未保留 PID、运行时长和命令信息
- 断线后运行方式：已确认 `nohup` 可用，本次未启动常驻任务

## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

### `gpustat`

- 安装版本：1.1.1
- Exit code：`1`
- 状态类别：NVML 或驱动不可用

### 状态解释

`nvidia-smi` 是 NVIDIA 驱动工具链提供的系统级程序。退出码 127 表明 shell 无法找到命令入口，问题发生在系统工具层；该结果不同于“已找到命令但未检测到设备”。

`gpustat` 是用户级 Python 工具。其安装成功只说明 Python 包及命令入口可用；实际查询仍依赖 NVIDIA 驱动、NVML 库和容器内可访问的设备。此次退出码 1 表明命令已经运行，但底层查询条件不具备。

两项命令均已真实执行并记录退出码。当前是 CPU-only 检查环境，因此不需要也不应通过安装系统级驱动或修改共享环境来强行使 GPU 检查成功。

## 飞书补充文档

- https://fudan-nlp.feishu.cn/docx/R0qAdO49yoB2TgxCerscCNZ5n5g

## 问题与收获

1. 区分“命令不存在”和“命令存在但底层设备查询失败”，避免混淆不同依赖层级的问题。
2. 使用临时 virtual environment 安装 `gpustat`，验证完成后清理，避免污染既有 Python 环境。
3. 权限测试只创建空模拟文件并设置 mode `600`，没有把真实凭据写入文件或日志。
4. 公开报告只保留退出码、状态类别和必要解释，不保留与核验无关的环境元数据。
