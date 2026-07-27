"""A2-P NVTX 阶段标记说明与稳定名称。

benchmark 实现使用这些同名区间把 kernel 映射回训练语义。独立常量模块便于
审查者确认标记口径，也避免报告、运行器与代码各写一套名称。
"""

PROFILE_WARMUP = "profile/warmup"
PROFILE_MEASURE = "profile/measure"
FORWARD = "forward"
BACKWARD = "backward"
OPTIMIZER_STEP = "optimizer_step"
ATTENTION_SCORES = "attention/scores"
ATTENTION_SOFTMAX = "attention/softmax"
ATTENTION_VALUE = "attention/value"

REQUIRED_RANGES = (
    PROFILE_WARMUP,
    PROFILE_MEASURE,
    FORWARD,
    BACKWARD,
    OPTIMIZER_STEP,
    ATTENTION_SCORES,
    ATTENTION_SOFTMAX,
    ATTENTION_VALUE,
)
