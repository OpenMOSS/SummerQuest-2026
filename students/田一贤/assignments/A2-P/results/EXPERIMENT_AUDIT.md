# A2-P 实验完整性复核

复核日期：2026-08-11。该报告由独立复核轮次基于提交目录中的文件重建判断；不是“49/49”式总分声明。

## 总结

结论：**WARN（可提交，但应保留限定语）**。

数值、CSV/JSON 结构、派生脚本和 H200 任务归属目前相互一致。A2-P 的 FP32/BF16 阶段 profiling 汇总均从远端原始 Chrome trace 重建，raw trace 的 SHA-256 保存在各自 `run_metadata.json`。仍有三项证据边界不能省略：原始 trace/snapshot 未随提交上传；独立 `nvidia-smi` probe 因当时无可调度 H200 容量而停止；A2-P 四个 shard 的 GPU ordinal 未被执行 wrapper 保留。因此不宣称 Nsight steady-state 归因、整卡 nvidia-smi 峰值或多 GPU speedup。

## 分项结论

| 项目 | 状态 | 依据 |
| --- | --- | --- |
| A：来源、ground truth、proxy | WARN | `A2-P/results/h200_provenance.json`、`A2-K/results/h200_sharding.json`；所有 proxy/reference 已显式标注，但 A2-P shard ordinal 只保留为未记录。 |
| B：统计与归一化 | PASS | `A2-P/results/benchmark.csv` 的 raw samples/mean/stdev/CV 与 `submission/profiling/common.py` 一致。 |
| C：结果与数字一致性 | PASS | FP32/BF16 profiling CSV 已为结构化 measured-from-raw-trace；README 只描述实际存在的 aggregate duration 字段。 |
| D：代码-产物链路 | PASS | `trace_summarize.py`、`summarize.py` 能生成当前 profiling 与 memory aggregate 格式。 |
| E：范围与证据 | WARN | profiling 限定为一次 framework-level `torch.profiler` step；A2-P exact GPU0–GPU3 mapping 未保留。 |
| F：evaluation type | PASS | timing/profiling/memory=`self_supervised_proxy`；nvidia-smi probe 明确未完成且不参与结果。 |

## 可核验 H200 provenance

qzcli API 核验到 A2-P 4×H200 成功任务，owner 为“田一贤”、project 为“前沿课题探索”。A2-P 的 exact shared artifact、派生 artifact 和 raw trace SHA-256 均记录在 `h200_provenance.json`。

## 保留的限制

- raw Chrome trace 和 memory-history pickle 只保留 hash，不在提交目录中；
- 独立 `nvidia-smi` 现场 probe 状态为 `stopped_unschedulable_no_h200_capacity`；
- A2-P profiling 是一次带一个 warm-up forward 的 framework-level trace，不是 Nsight steady-state kernel 统计；
- A2-P 四个 shard 的 GPU ordinal 未被 wrapper 写入提交 artifact；memory snapshot 脚本未显式设置 seed；
