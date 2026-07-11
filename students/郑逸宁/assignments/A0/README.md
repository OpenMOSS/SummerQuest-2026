# A0 公开提交：郑逸宁

> 这是 fork 内为 A1 端到端流程测试准备的最小有效基线，不作为正式 A0 作业评分材料。

## GitHub 与 PR

- 分支：`test/a1-basics-release`
- Git 操作总结：已完成 fork、分支和基线提交准备

## Linux 环境摘要

- 操作系统：Ubuntu 22.04
- Python：Python 3.12
- Virtual environment：已创建
- 模拟密钥文件权限：仅所有者可读写
- 常驻进程方式：tmux

## GPU 状态检查

### `nvidia-smi`

- Exit code：0
- 状态类别：命令可执行

### `gpustat`

- 安装版本：已安装
- Exit code：0
- 状态类别：命令可执行

### 状态解释

本文件只用于建立结构有效的测试基线；不公开主机名、设备编号、进程、内部路径或完整命令输出。

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/LTvWdubEko6syix0BU7czH5Pnbg

该文档设置为组织内公开，用于说明本次 A1 流程测试。

## 问题与收获

1. 学生 A1 PR 必须只修改本人单个 A1 目录。
2. 外部 `assignment1-basics` 仓库应与课程仓库保持平级。

## 自检

- [x] 公开内容未包含用户名、主机名、IP、内部路径、进程参数和组内数据。
- [x] GitHub 和飞书正文均未包含 Secret、Token、Cookie、密码或私钥。
- [x] 飞书补充文档已设置为组织内公开。
