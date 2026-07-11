# A1 公开提交：郑逸宁

> 这是用于验证作业提交与后续评分流程的测试版非满分提交。代码公开可见，未包含访问凭据、内部路径、私有数据或模型权重。

## 基本信息

- 作业题面版本：26.0.3
- 完成范围：Linear、Embedding、SiLU、Softmax、Cross Entropy、Gradient Clipping
- 未完成项：Tokenizer、BPE、RMSNorm、RoPE、Attention、Transformer、AdamW、学习率调度、数据采样与 checkpoint
- 上游 starter commit：`a158843b20107949f1a8d7df1b05cd33b9166712`
- 本地工作仓库：`../assignment1-basics`（与 `SummerQuest-2026` 同级）

## Markdown 报告

本提交有意只完成六个基础接口，用于测试评分系统能否区分已完成与未完成项目。已完成接口统一在 `partial_submission.py` 中实现，由 `tests/adapters.py` 做参数转接；其他 adapter 保持 starter 中的 `NotImplementedError`，因此失败项是确定且可解释的。

完整公开测试共收集 48 项：6 项通过、41 项失败、1 项预期失败。失败均来自明确未实现的 adapter，不依赖随机制造的错误答案。

## 复现说明

- 环境与依赖：test991、Python 3.12.12、uv、CPU 环境
- 数据准备：公开测试 fixtures 由原始仓库自带，无额外数据
- 目标测试命令：`uv run pytest -q tests/test_model.py::test_linear tests/test_model.py::test_embedding tests/test_model.py::test_silu_matches_pytorch tests/test_nn_utils.py`
- 完整测试命令：`uv run pytest -q --tb=no`
- 同步命令：`python3 scripts/sync_a1_submission.py --name '郑逸宁'`
- 配置文件：无

## 代码与脚本

- 真实实现：`submission/cs336_basics/partial_submission.py`
- 测试 adapter：`submission/tests/adapters.py`
- 训练、数据编码与生成脚本：无
- 实现说明：测试夹具使用 PyTorch 官方算子实现六个基础接口，重点验证提交流程和评分边界，不作为完整 A1 解答。

真实实现先在兄弟目录 `../assignment1-basics` 中完成并通过对应官方测试，再使用同步命令复制到本目录。未提交公共 tests、fixtures、数据、模型权重、虚拟环境或依赖锁。

## 实验日志

- 日志目录：`logs/`
- 文件与格式：`logs/public-tests.txt`，纯文本记录命令、环境与 pytest 汇总
- 与报告中实验的对应说明：日志分别记录六个目标测试全通过和完整套件的非满分结果

## 飞书补充文档

- 链接：https://fudan-nlp.feishu.cn/docx/LTvWdubEko6syix0BU7czH5Pnbg

该文档设置为组织内公开，仅说明本次流程测试，不保存密钥、访问凭据或内部材料。
