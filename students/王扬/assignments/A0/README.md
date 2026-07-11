# A0 公开提交：王扬

## GitHub 与 PR

- 分支：`a0/YangW796`
- Git 操作总结：fork、upstream、branch、commit、push、PR 已完成

## Linux 环境摘要

- 操作系统：22.04.5 LTS (Jammy Jellyfish)
- Python：3.13.11
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功
```text
Sat Jul 11 06:33:42 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 570.124.06             Driver Version: 570.124.06     CUDA Version: 12.8     |
|-----------------------------------------+------------------------+----------------------+
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
summer-quest-01  Sat Jul 11 06:33:26 2026  570.124.06
[0] NVIDIA xxx 80GB HBM3 | 32°C,   0 % |     0 / 81559 MB |
```

### 状态解释

退出码 0：命令执行成功，检测到 GPU 并可正常访问
退出码 1：命令执行失败（可能原因：无 NVIDIA 驱动、无 GPU 设备、NVML 不可用）
退出码 127：命令未找到（nvidia-smi 或 gpustat 不存在）
说明：在 CPU 服务器上，退出码非 0 是正常现象，不影响得分


## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/WgCDdzATZoI1bqxhnpGcuLssn8f?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

### 1. Git 分支管理——上游同步冲突

**问题**：在 Fork 的仓库中创建作业分支时，发现本地 main 分支落后于上游仓库多个 commit，直接创建分支会导致作业基于旧版本代码，后续 PR 可能出现大量冲突。

**排查**：
- `git log --oneline --graph --all` 查看提交历史，发现本地 main 与 upstream/main 分叉
- `git remote -v` 确认 upstream 已正确添加
- `git branch -vv` 查看本地分支跟踪的远程分支状态

**解决**：
```bash
git checkout main
git fetch upstream
git merge upstream/main   # 合并上游最新代码
git push origin main      # 同步到自己的 Fork
```

**收获**：在 Fork 协作中，每次创建新分支前应先同步上游仓库，养成 `fetch → merge/rebase` 的习惯。使用 `git pull upstream main` 是 `fetch + merge` 的快捷方式，但显式使用 `fetch` + `merge` 更可控。

---

### 2. uv 虚拟环境——`uv pip` 与系统 Python 的隔离

**问题**：使用 `uv pip install gpustat` 后发现 `gpustat` 命令在终端中找不到（`command not found`），但在 `uv run gpustat` 下可以执行，原因不明。

**排查**：
- `which python` 确认当前 Python 路径
- `echo $PATH` 查看环境变量，发现虚拟环境的 bin 目录未加载
- `uv pip list` 确认 gpustat 已安装到虚拟环境

**解决**：
```bash
# 方式一：激活虚拟环境后再执行
source .venv/bin/activate
gpustat

# 方式二：使用 uv run 自动在虚拟环境中执行
uv run gpustat
```

**收获**：`uv pip install` 默认安装到当前激活的虚拟环境，但 `uv run` 会在隔离环境中执行命令，无需手动激活。理解 `uv` 的"工具"与"环境"概念区分很重要——`uv` 管理 Python 版本和虚拟环境，`uv pip` 对标传统 `pip`，而 `uv run` 是执行入口。

---

### 3. 退出码判断——`$?` 的时机敏感性

**问题**：执行 `nvidia-smi` 后立即用 `echo $?` 显示退出码为 0，但记录到文档时总写成 127，数据不一致。

**排查**：发现是在执行其他命令（如 `ls` 或 `echo`）之后才执行 `echo $?`，此时 `$?` 已被覆盖为最后一条命令的退出码。

**解决**：
```bash
nvidia-smi
EXIT_CODE=$?   # 立即保存到变量
echo "nvidia-smi 退出码: $EXIT_CODE"
# 中间可以执行其他命令，不影响已保存的变量
```

**收获**：`$?` 只保留**上一条命令**的退出码，任何后续命令（包括 `echo` 本身）都会改变它。需要记录退出码时应立即存入变量或写入文件。退出码 0 表示成功，非 0 表示失败，128+ 通常由信号引起（如 127 = command not found，130 = Ctrl+C 终止）。

---

### 4. 文件权限——`chmod 600` 的目录适用性

**问题**：尝试对 `~/workspace/a0` 目录执行 `chmod 600`，期望限制访问，但发现进入该目录时提示权限不足（`Permission denied`），连 `cd` 都无法执行。

**排查**：`ls -ld ~/workspace/a0` 显示权限为 `d------`（目录权限 600），目录缺少执行权限（x），导致无法进入。

**解决**：
- 对于目录，需要有执行权限（x）才能进入
- `chmod 700 ~/workspace/a0` 比 `600` 更适合目录（所有者有 rwx）
- 对于敏感配置文件使用 `chmod 600`，目录使用 `chmod 700`

**收获**：Linux 权限中，目录的"执行"权限代表"进入"权限。`600` 适用于**文件**（所有者可读写），`700` 适用于**目录**（所有者可读写执行）。`chmod 600` 对文件是正确的，对目录会阻断访问。理解 rwx 在不同对象上的含义差异是 Linux 权限管理的基础。

---

### 5. uv 配置私有 PyPI 源

**问题**：在安装 gpustat 时，如果使用公司私有 PyPI 源，需要配置 index-url，但 uv 的配置方式与 pip 不同。

**解决**：
```bash
# 方式一：环境变量（临时）
export UV_INDEX_URL="xxx"

# 方式二：uv 配置文件（永久）
# 创建 uv.toml

[pip]
index-url = "xxx"
allow-insecure-host = ["xxx"]
```

**收获**：uv 不支持 `pip config set` 命令，需通过环境变量或 `uv.toml` 配置。HTTP 源需额外配置 `insecure-host` 允许非 HTTPS 连接。`uv pip install --index-url` 也可临时指定源，但推荐使用配置文件统一管理。也可以现在cpu机器上装好，再在gpu机器上使用。



## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
