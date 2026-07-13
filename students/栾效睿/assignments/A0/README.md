# A0 公开提交：栾效睿

## GitHub 与 PR

- 分支：`a0/BraveBeter`
- Git 操作总结： 已将课程仓库fork到个人仓库，使用`git clone`命令克隆到个人服务器，使用`git remote add`命令添加课程仓库地址为`upstream`，使用`git pull upstream main`命令将服务器main分支更新为最新，使用`git checkout -b` 命令切换分支到`a0/BraveBeter`

## Linux 环境摘要

- 操作系统：Linux 5.15.0-119-generic x86_64(Ubuntu 22.04.4 LTS) 
- Python：3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：nohup、systemd


## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在

```text
printf 'gpustat exit_code=%s\n' "$gpustat_exit_code"
bash: nvidia-smi: command not found
nvidia-smi exit_code=127
Error on querying NVIDIA devices. Use --debug flag to see more details.
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Driver Not Loaded
gpustat exit_code=1
```

### 状态解释

| 命令 | 执行结果 | 退出码 | 成功或失败的原因 | 主要依赖 |
| --- | --- | :---: | --- | --- |
| `nvidia-smi` | 执行失败，提示 `command not found` | `127` | 系统没有在当前 `PATH` 中找到 `nvidia-smi` 可执行文件，因此命令尚未进入查询 GPU 或驱动状态的阶段。 | `nvidia-smi` 用户态工具、NVIDIA GPU、匹配且已加载的 NVIDIA 内核驱动、设备访问权限 |
| `gpustat` | 程序能够启动，但查询 GPU 失败，提示 `Driver Not Loaded` | `1` | Python 包已经正确安装，但程序无法通过 NVML 连接到可用的 NVIDIA 驱动或设备，因此无法返回 GPU 状态。 | Python 虚拟环境、NVML、已加载的 NVIDIA 驱动和可访问的 GPU |

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/IOeywQEddi8O3pkleTYcXjLanTf?from=from_copylink
 

## 问题与收获

- 使用 `python -m venv`命令创建虚拟环境失败， 安装了`uv`并转用`uv venv` 完成虚拟环境创建。
- 使用`nohup python run_task.py > run.log 2>&1 &`命令运行，运行了几分钟之后使用`kill PID`结束程序，结果`cat run.log`没有预期的输出信息。因为print的内容一直在Block Buffer攒够特定大小（通常4KB）才会写，直接kill进程，信息没攒够，未写入文件。使用`nohup python -u`即可解决，会强制刷新写入不等待缓冲区满。

## 自检

- [x] 我实际运行了 nvidia-smi 和 gpustat，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 sudo 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。