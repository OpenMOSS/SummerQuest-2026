# A0 公开提交：陈匡巍

> 本文件公开可见，只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/alpacaking`
- Git 操作总结：已将官方仓库 Fork 到个人 GitHub 账号，并将个人 Fork 配置为 `origin`、官方仓库配置为 `upstream`。本地 `main` 已同步最新 `upstream/main`，随后创建独立分支 `a0/alpacaking` 完成 A0。本次修改范围限制在 `students/陈匡巍/` 目录内；后续将使用 Conventional Commits 提交、push 到个人 Fork，并向上游仓库创建 PR。

## Linux 环境摘要

- 操作系统：Ubuntu 24.04 LTS，Linux x86_64。
- Python：Python 3.12.3。
- Virtual environment：已在用户级环境中创建并启用，不使用 `sudo pip`。
- 模拟密钥文件权限：已设置为 `600`，仅当前用户可读写。
- 常驻进程方式：已了解并检查当前用户进程；后续需要长时间任务时优先使用 `tmux` 保持 SSH 断开后的会话。

公开摘要已删除用户名、主机名、IP、内部路径、SSH 配置、完整进程参数和硬件容量信息。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功。

命令能够返回设备管理接口状态。公开报告不展示具体设备型号、数量、UUID、利用率、进程区或服务器容量。

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功。

命令能够在当前 Python virtual environment 中正常启动并返回设备状态摘要。公开报告不展示主机名、具体设备型号、数量、利用率、进程区或服务器容量。

### 状态解释

`nvidia-smi` 是系统层面的设备管理工具，本次退出码为 0，说明当前环境可以访问对应的设备管理接口。`gpustat` 是依赖 Python 包和底层管理库的上层状态查看工具，本次在用户级 virtual environment 中安装后退出码也为 0，说明 Python 环境能够正常调用相关接口。两项检查均只记录退出码和状态类别，没有为了通过检查而使用 `sudo` 安装驱动或修改系统环境。

## 飞书补充文档

- 链接：https://lako5livxd0.feishu.cn/wiki/Y2cIw8TNGioGcek6RImcJPNdnre?from=navigation

该文档设置为组织内公开，用于保存 A0 的组内验收材料，未开启互联网公开访问。文档正文不保存 Secret、Token、Cookie、密码或私钥。

## 问题与收获

- 完成了从 Fork、配置 `upstream`、同步主分支到创建独立 A0 分支的 GitHub 协作流程。
- 练习了 Linux 基础环境检查、Python virtual environment 创建、用户级包安装和敏感配置文件权限设置。
- 实际运行了 `nvidia-smi` 与 `gpustat`，理解了系统工具和 Python 上层工具都可能受到驱动、设备管理库和运行环境影响。
- 明确了公开 GitHub 报告和组内飞书材料的边界：公开材料只写脱敏摘要，内部路径、主机名、进程、硬件明细和凭据都不能写入 GitHub。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
