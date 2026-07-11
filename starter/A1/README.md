# A1 公共 starter

本目录是 A1 的共享开发与公共测试环境。它基于
[`stanford-cs336/assignment1-basics`](https://github.com/stanford-cs336/assignment1-basics)
commit `a158843b20107949f1a8d7df1b05cd33b9166712`，上游代码和测试按本目录中的
[`LICENSE`](LICENSE) 使用。

学生不要直接修改公共 starter，也不要把整个 starter 放入个人提交目录或 PR。先从仓库
根目录运行：

```bash
python3 scripts/create_assignment.py --name '<同学真名>' --assignment A1
```

脚手架只会复制需要作答的 `cs336_basics/` 和 `tests/adapters.py`，避免每位同学重复提交
公共 tests、fixtures 和依赖锁。

## 环境

A1 使用 Python 3.12 或 3.13，并由 `uv.lock` 固定公共依赖。

不要在个人提交中添加或修改依赖文件。如果实现确实需要新的第三方依赖，请先由课程
维护者统一更新本目录。

## 公共测试

本目录保留原作业的测试布局和运行方式。不要直接修改仓库中的公共 starter；在仓库外
创建个人工作副本：

```bash
cp -R starter/A1 ../a1-work
cd ../a1-work
uv sync --frozen
uv run pytest
```

在工作副本中实现 `cs336_basics/` 并填写 `tests/adapters.py`。完成后，只把正式题面列出的
个人实现、adapter、脚本、Markdown `README.md` 和日志复制到
`students/<同学真名>/assignments/A1/`；不要提交公共 tests、fixtures、`pyproject.toml`
或 `uv.lock`。21 个固定 adapter 的签名见 [`tests/adapters.py`](tests/adapters.py)。

公共测试覆盖核心 tokenizer、Transformer 和训练工具接口；训练脚本、生成、完整实验和
书面分析按[正式题面](../../assignments/A1/README.md)完成。

实验日志是必交材料并统一放在个人 A1 目录的 `logs/`；具体文件、格式、字段和评分规则
由作业批改助教完善，不属于本次固定的 21 个 adapter 接口。

## 实验数据

在个人工作副本中创建 `data/` 并下载数据；starter 自带的 `.gitignore` 会忽略该目录：

```bash
mkdir -p data
cd data

wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-train.txt
wget https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_train.txt.gz
gunzip owt_train.txt.gz
wget https://huggingface.co/datasets/stanford-cs336/owt-sample/resolve/main/owt_valid.txt.gz
gunzip owt_valid.txt.gz
```
