# A0 公开提交：田一贤

> 本文件公开可见，只记录属于本人且可以核验的脱敏信息。服务器账号、主机名、IP、内部路径、完整进程参数和任何凭据均不写入公开仓库。

## 当前完成状态

- GitHub Fork 与远端配置：已核对当前工作区配置
- 公开 Profile：已按本人经历完成
- Linux 环境与 GPU 检查：已在个人 Linux 服务器实际执行并记录结果
- 飞书个人主页与 A0 补充文档：需要补充本人链接并确认权限

当前版本已删除原报告中无法追溯到本人操作记录的环境数据，只保留本次实际执行并复核过的结果。

## GitHub 与 PR

- GitHub ID：`tianyixian`
- 计划使用的 A0 提交分支：`a0/tianyixian`
- 当前已核对的仓库配置：个人 Fork 为 `origin`，课程仓库为 `upstream`
- 提交流程：同步最新 `upstream/main` 后创建独立 A0 分支，只提交 `students/田一贤/` 下的本人文件；使用 Conventional Commits 提交并推送到个人 Fork，再向上游 `main` 创建 PR
- PR 链接：不在报告中维护；以 GitHub 上的实际 PR 记录为准

## Linux 环境摘要

以下结果来自本次在本人的个人服务器上的实际检查：

- 操作系统：Ubuntu 22.04.5 LTS（Linux x86_64）
- Python 版本：3.12.11
- Virtual environment：已创建用户级 Python virtual environment，并在其中安装 `gpustat 1.1.1`
- 模拟敏感文件权限：`600`，已通过 `stat` 复核
- 常驻进程方式：`tmux 3.2a`；已创建 detached session，并通过 `tmux has-session` 验证（Exit code `0`）

公开报告只填写操作系统类型、Python 版本和操作结论；用户名、主机名、IP、内部路径、硬件容量、SSH 配置及完整进程参数应继续保留在公开文件之外。

## GPU 状态检查

### `nvidia-smi`

- Exit code：`0`
- 状态类别：命令执行成功

### `gpustat`

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 共享库不可用

### 复核命令

在本人用户级 virtual environment 中执行并保存退出码：

```bash
set +e

nvidia-smi
nvidia_smi_exit_code=$?
printf 'nvidia-smi exit_code=%s\n' "$nvidia_smi_exit_code"

gpustat
gpustat_exit_code=$?
printf 'gpustat exit_code=%s\n' "$gpustat_exit_code"
```

### 结果解释

- 退出码为 `0` 时，只能说明本次命令成功完成；公开报告仍需隐藏 GPU 型号、数量、UUID、利用率和进程信息。
- `nvidia-smi` 返回 `127` 通常表示 shell 找不到命令，不能据此断定物理服务器一定没有 GPU。
- `gpustat` 返回非零且提示 NVML 或驱动不可用时，只能说明当前查询链路不可用；不能把它写成已经确认“没有 GPU”。
- 不得为了让检查成功而使用 `sudo` 安装或修改系统级 NVIDIA 驱动。

本次 `nvidia-smi` 返回 `0`，说明命令在该环境中成功执行；公开报告不保留设备型号、UUID、利用率和进程明细。`gpustat` 已在用户级 virtual environment 中安装成功，但查询阶段返回 `1` 并提示 NVML 共享库不可用，因此只能判断 Python 工具的 GPU 查询链路不可用，不能据此断言物理设备不存在。本次未使用 `sudo` 安装或修改系统级驱动。

## 飞书补充文档

- 链接：尚未提供本人的 A0 飞书补充文档链接
- 权限状态：尚待确认组织内公开，并关闭互联网公开访问

飞书补充文档只保存助教核验所需的最小脱敏输出，包括 Linux 环境检查、`nvidia-smi` 与 `gpustat` 的关键结果、退出码和排查结论；不得保存 Secret、Token、Cookie、密码、私钥或完整内部环境信息。

## 问题与收获

- 模板和其他同学的报告只能用于了解提交结构，不能作为本人环境、退出码或完成状态的证据。
- GitHub 报告中的每项完成声明都应能回到本人的命令记录或仓库状态；无法核验的内容应先明确标记，再重新执行补齐。
- `origin` 用于个人 Fork，`upstream` 用于同步课程仓库；每次作业应从最新 `upstream/main` 创建独立分支。
- 判断 GPU 状态时需要结合命令是否存在、退出码和错误信息，区分“命令不存在”“驱动或 NVML 不可用”和“查询成功”。
- 公开 GitHub 只保留脱敏摘要，组织内差量证据放入飞书，任何凭据都不进入两类文档。

## 完成前自检

- [x] 报告中的姓名、GitHub ID 和分支命名均属于本人。
- [x] 当前文件没有沿用他人的 PR、飞书链接或环境结果。
- [x] 我已在本人的 Linux 环境中实际运行 `nvidia-smi` 和 `gpustat`，并填写真实 Exit code。
- [x] 我已补充本人的 Linux、Python、virtual environment、权限和常驻进程检查结论。
- [ ] 我已创建本人的 A0 飞书补充文档，并确认组织内公开且未开启互联网公开访问。
- [x] 当前 GitHub 文件不包含用户名、主机名、IP、内部路径、完整进程参数、Secret、Token、Cookie、密码或私钥。
