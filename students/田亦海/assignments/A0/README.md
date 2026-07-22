# A0 公开提交：田亦海

## GitHub 与 PR

- 分支：`a0/limboy058`
- Git 操作总结：已完成仓库 Fork 和 clone，将实验室仓库配置为 `upstream`，并从最新的 `upstream/main` 创建 `a0/limboy058` 分支；修改完成后使用 Conventional Commits 提交、推送到个人仓库并创建 PR。

## Linux 环境摘要

- 操作系统：Linux（Ubuntu 24.04）
- Python：3.12.3
- Virtual environment：已创建并完成激活与 `pip` 升级
- 模拟密钥文件权限：`600`，仅当前用户可读写
- 常驻进程方式：`tmux`

## GPU 状态检查

### nvidia-smi

- Exit code：`0`
- 状态类别：命令执行成功，但未返回任何输出

### gpustat

- 安装版本：`1.1.1`
- Exit code：`1`
- 状态类别：NVML 或驱动不可用

```text
Error on querying NVIDIA devices. Use --debug flag to see more details.
NVML Shared Library Not Found
```

### 状态解释

`nvidia-smi` 在当前环境中存在且退出码为 0，但没有输出 GPU 或驱动状态，是因为cpu服务器加载了之前gpu服务器上保存的镜像. 所以有nvidia-smi软件.

`gpustat` 是 Python 层的状态查看工具，底层仍依赖 NVIDIA 的 NVML 共享库；由于cpu服务器没有相关库所以报错. 不过这不一定代表物理机器没有 GPU，也可能是未安装驱动、驱动未加载，或容器未挂载 GPU 和驱动库。若 NVIDIA GPU、驱动及环境配置均正常，gpustat 通常可以正常显示状态。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/MJzpw8NF4iFerTkVTcGcL79SnLd

## 问题与收获

- 使用用户级 virtual environment 可以隔离 Python 包，平时在项目开发时我一般使用uv来管理环境.
- 敏感配置文件应使用 `600` 权限，避免被其他用户读取或修改。
- `gpustat` 依赖 NVML。NVML 不可用时，只能说明当前无法检查 GPU，不能直接断言机器没有 GPU。

## 自检

- [✓] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [✓] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [✓] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [✓] GitHub 正文没有任何 Secret、Token、Cookie、密码或私钥；飞书正文待任务 4 完成时检查。
- [✓] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
