# A0 公开提交：王群超


## GitHub 与 PR

- 分支：`a0/bhuj209`
- Git 操作总结：fork、upstream、branch、commit、push、PR操作全部完成

## Linux 环境摘要

- 操作系统：Ubuntu 22.04.4 LTS
- Python：Python 3.10.12
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：tmux



## GPU 状态检查

### `nvidia-smi`

- Exit code：127
- 状态类别：命令不存在


### `gpustat`

- 安装版本：1.1.1
- Exit code：1
- 状态类别：驱动不可用


### 状态解释

nvidia-smi是因为系统中未安装该命令，即当前服务器节点没有NVIDIA显卡驱动管理工具。
gpustat需要调用NVIDIA驱动提供的NVML（NVIDIA Management Library）接口来查询 GPU 状态，而当前节点没有加载 NVIDIA 内核驱动。

## 飞书补充文档

无补充内容

## 问题与收获

1. 知道了怎么创建虚拟环境。
2. 了解什么是常驻进程方式，以及怎么样使用tmux。
3. 熟悉了git的使用。


## 自检

- [✅️] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [✅️] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [✅️] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [✅️] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [✅️] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
