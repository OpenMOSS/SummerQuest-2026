# A0 公开提交：杭瑞文

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/RavenHang`
- Git 操作总结：fork、upstream、branch、commit、push、PR 全部完成

## Linux 环境摘要

- 操作系统：Ubuntu 24.04.2 LTS
- Python：Python 3.12.3
- Virtual environment：openmoss-a0
- 模拟密钥文件权限：600 ～/openmoss-a0/example.env
- 常驻进程方式：tmux,nohup,screen

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
nvidia-smi
Sat Jul 11 13:35:34 2026       
+-----------------------------------------------------------------------------------------+
| NVIDIA-SMI 550.163.01             Driver Version: 550.163.01     CUDA Version: 12.4     |
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
# gpu stat
Name: gpustat
Version: 1.1.1
Summary: An utility to monitor NVIDIA GPU status and usage
License: MIT
Location: ～/.venvs/openmoss-a0/lib/python3.12/site-packages
Requires: blessed, nvidia-ml-py, psutil
Required-by: 
```

### 状态解释

由于使用了启智平台的 GPU 计算类型组，基础镜像为：pytorch:25.06-py3:25.06 已预装 NVIDIA 驱动，故 nvidia-smi 成功。

gpustat 安装在虚拟环境中，并且已安装其所需的依赖包，该工具本质上是对 nvidia-smi 信息的二次封装，但它需要额外的 Python 库支持，故 gpustat 成功。
## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/DMOYwS2TKiPJd0ks9AccMNYIn8g?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获
无遇到问题，学习到了基础 GPU 环境配置与校验。
## 自检

- [√] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [√] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [√] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [√] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [√] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
