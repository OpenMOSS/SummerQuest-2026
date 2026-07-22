# A0 公开提交：龚天时

> 本文件公开可见。只写脱敏结果；不能公开但确有审核必要的材料放在下方链接的飞书补充文档中。

## GitHub 与 PR

- 分支：`a0/Tianshi-Gong1002`
- Git 操作总结：
    已完成课程仓库 Fork。

    已将个人 Fork clone 到本地。

    已将课程官方仓库添加为 upstream。

    已从官方最新 main 创建个人 A0 分支。

    已使用 Conventional Commits 格式完成提交。

    已将个人分支 push 至个人 Fork。

    Pull Request：待完成其余材料后创建。



## Linux 环境摘要

- 操作系统："Ubuntu 24.04.2 LTS"
- Python：Python 3.12.3
- Virtual environment：已创建
- 模拟密钥文件权限：600
- 常驻进程方式：nohup

不要填写用户名、主机名、IP、内部路径、SSH 配置或完整进程参数。

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：成功

```text
+-----------------------------------------------------------------------------------------+
| Processes:                                                                              |
|  GPU   GI   CI              PID   Type   Process name                        GPU Memory |
|        ID   ID                                                               Usage      |
|=========================================================================================|
|  No running processes found                                                             |
+-----------------------------------------------------------------------------------------+
nvidia-smi exit code: 0
```

### `gpustat`

- 安装版本：1.1.1
- Exit code：0
- 状态类别：成功

```text
[0] **********. | 29°C,   0 % |    16 / 49140 MB |
gpustat exit code: 0

Name: gpustat
Version: 1.1.1
Summary: An utility to monitor NVIDIA GPU status and usage
```

### 状态解释

两个命令都成功，nvidia-smi成功依赖正确分配的NVIDIA GPU，以及可用的驱动环境。
gpustat需要现在python 环境中安装，因为选择了可联网pgu资源，可以正常运行安装。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/wiki/PwYBwHUNciPR9ekT5jCcXRLyn2b?from=from_copylink

该文档设置为组织内公开，用于保存 A0 的组内验收材料。

## 问题与收获

1. 多 GitHub 账号需要通过独立 SSH key 和 Host alias 区分。
2. `origin` 用于个人 Fork，`upstream` 用于同步官方仓库。
3. 公钥可以交给 GitHub 或服务器，私钥必须只保留在本地。
4. 分布式训练实例无法访问外部软件源，导致 `gpustat` 和 Git 安装受阻。后续转到可联网gpu解决。
5. 公开仓库不能出现主机名、内部路径、GPU 资源细节或进程信息。

## 自检

- [x] 我实际运行了 `nvidia-smi` 和 `gpustat`，并记录了退出码。
- [x] 我没有为了 GPU 检查使用 `sudo` 安装驱动或修改系统环境。
- [x] 公开内容已删除用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文都没有任何 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开，且没有开启互联网公开访问。
