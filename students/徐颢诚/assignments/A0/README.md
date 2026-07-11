# A0 公开提交：haochengxu

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/VegTea`
- Git 操作总结：
    1. 首先在github上fork了团队的仓库

    2. 然后git clone刚刚fork的仓库`git clone https://github.com/VegTea/SummerQuest-2026.git`

    3. 把上游仓库设置为团队的仓库`git remote add upstream https://github.com/OpenMOSS/SummerQuest-2026.git`

    4. 从主分支创建分支a0/VegTea `git checkout -b a0/VegTea upstream/main`

    5. 然后开始写a0的作业

    6. git commit和push到自己fork的仓库，然后提交PR

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：3.13.9
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux

## GPU 状态检查

### `nvidia-smi`

```
nvidia-smi
Exit code 127
command not found: nvidia-smi
```

### `gpustat`

```
gpustat
Exit code 1
Error on querying NVIDIA devices.
```

### 状态解释

因为我用的是启智的CPU服务器,没有GPU所以两个命令都找不到GPU. nvidia-smi依赖于 NVIDIA 内核驱动模块（nvidia.ko）。物理机必须插了 NVIDIA GPU 并安装了对应驱动后才能使用. gpustat依赖于python环境和nvidia驱动以及NVML共享库

## 飞书补充文档

无补充

## 问题与收获

掌握标准开发流程

## 自检

- [✅] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [✅] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [✅] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [✅] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [✅] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
