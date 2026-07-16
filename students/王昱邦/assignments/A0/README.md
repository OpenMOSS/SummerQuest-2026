# A0 公开提交：王昱邦

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/No-518`
- Git 操作总结：已将课程官方仓库 Fork 到个人 GitHub 账号，并将个人 Fork 配置为 `origin`、课程官方仓库配置为 `upstream`。在确认本地 `main` 与最新 `upstream/main` 一致后，创建独立分支 `a0/No-518` 完成 A0，修改范围仅限个人学生目录。提交前已检查差异并运行仓库校验，使用 Conventional Commit 创建 commit，推送到个人 Fork，并向课程官方仓库的 `main` 分支创建了 A0 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.5 LTS（Linux x86_64）
- Python：3.10.12
- Virtual environment：已创建（使用用户级 `virtualenv`）
- 模拟密钥文件权限：600
- 常驻进程方式：`tmux` 3.2a

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

### 状态解释

`nvidia-smi` 能够通过 NVIDIA 驱动读取 GPU 状态；`gpustat` 已安装在独立的 Python 虚拟环境中，并能通过 NVML 读取简化状态。两个命令的退出码都是 0，说明当前环境可以检测 GPU 并读取驱动状态；这不代表 GPU 当前空闲或一定满足具体训练任务的资源需求。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/PS63d4YrboF44gx01UjcPQ2rn5c

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 首次使用 `python3 -m venv` 创建虚拟环境时，系统因缺少 `ensurepip/python3-venv` 而失败。我没有使用 `sudo` 修改系统 Python，而是检查现有用户级工具，最终使用 `virtualenv` 成功创建隔离环境。
- 通过实际运行 `nvidia-smi` 和 `gpustat`，我理解了需要结合退出码和错误类别解释 GPU 状态；命令成功也不等于 GPU 空闲或一定满足具体训练需求。
- 在整理公开报告和飞书补充材料时，我进一步明确了 GitHub 公开内容、组内审核证据与不应进入任何文档的机密凭据之间的边界。
- 使用 Lark CLI 创建补充文档时，我理解了用户身份与应用身份在资源所有权、授权范围和文档管理能力上的差异。
- 本次教师示例在当前 Linux 机器上完成，没有单独验证 SSH 登录链路。如将 SSH 保留为学生正式 A0 的必做项，还需补充最小 SSH 登录验证。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
