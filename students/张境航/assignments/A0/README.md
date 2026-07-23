# A0 公开提交：张境航


## GitHub 与 PR

- 分支：`a0/Elvira-hang920`
- Git 操作总结：已完成课程仓库的 Fork、个人服务器上的 clone、`upstream` 配置和 A0 分支创建；已使用脚手架创建个人学生目录，并仅在个人目录中完成修改。完成检查后，将使用 Conventional Commits 提交、push 到个人 Fork，并向上游仓库创建 Pull Request。

## Linux 环境摘要

- 操作系统：Ubuntu 20.04 LTS
- Python：Python 3.8.10
- Virtual environment：已创建并成功激活用户级虚拟环境 `openmoss-a0`
- 模拟密钥文件权限：`600`
- 常驻进程方式：使用 `nohup` 配合后台执行符号 `&` 完成测试


## GPU 状态检查

### `nvidia-smi`

- Exit code：`127`
- 状态类别：命令不存在

```text
nvidia-smi: command not found
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：NVML或驱动不可用

```text
Error on querying NVIDIA devices.
Unknown Error
```

### 状态解释

`nvidia-smi` 是 NVIDIA 提供的系统级 GPU 管理与状态查询工具，依赖服务器已安装相应命令、NVIDIA 驱动，并且该命令位于当前环境的可执行路径中。当前登录环境无法找到 `nvidia-smi`，因此 shell 返回 `command not found`，退出码为 `127`。

`gpustat` 是安装在 Python 虚拟环境中的用户级工具，本身已经成功安装，但它需要通过 NVML（NVIDIA Management Library）读取 NVIDIA GPU 和驱动状态。执行时设备查询失败，因此退出码为 `1`。这两个结果只能说明当前登录环境无法完成 GPU 状态查询，不能据此判断整个 Slurm 集群中没有 GPU。

## 飞书补充文档

https://fudan-nlp.feishu.cn/wiki/YcO6wuTRTiQCq5kOsHvcI7kKngd?from=from_copylink

## 问题与收获

详见飞书补充文档

## 自检

- [ ] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [ ] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [ ] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [ ] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [ ] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
