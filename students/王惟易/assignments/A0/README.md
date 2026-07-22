# A0 公开提交：王惟易

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/Mtrya`
- Git 操作总结：在 GitHub 创建 fork，克隆 fork 到本地，并将课程仓库设为 upstream，然后在 upstream/main 的基础上创建分支 a0/Mtrya，完成 A0 作业后根据规范 commit，push 到 origin，最后创建 PR 指向官方课程仓库主分支。

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS, Linux x86_64
- Python：3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux 3.2a

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

### 状态解释

nvidia-smi 依赖正确的 NVIDIA 显卡驱动程序以及底层的 NVIDIA 管理库 (NVML)，本次运行成功，说明两者都没问题；gpustat 安装在个人目录下的虚拟环境中，依赖底层 NVML，同样正常返回。显卡正确识别。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/Ha0twKbWHiMx3Rk8NjqcYX40n6f

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

- 普通文件的权限是 644，所有者可以读写，其他人只读，使用`chmod 600`命令以后权限变为 600，所有者可以读写，同组和其他人不能读也不能写
- 使用虚拟环境隔离项目 Python 环境和系统 Python 环境

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。