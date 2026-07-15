"""可组合的实验日志后端。"""

from __future__ import annotations

import json
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any


class ExperimentLogger(ABC):
    """训练代码使用的统一日志接口。"""

    @abstractmethod
    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        """记录一个训练或验证指标点。"""

    @abstractmethod
    def log_summary(self, summary: Mapping[str, Any]) -> None:
        """记录一次 run 的最终汇总信息。"""

    def close(self) -> None:
        """释放可选后端占用的资源。"""


class JsonlLogger(ExperimentLogger):
    """将逐点指标写入 JSONL，将汇总写入 summary.json。"""

    def __init__(self, run_directory: Path) -> None:
        run_directory.mkdir(parents=True, exist_ok=True)
        self.metrics_file = (run_directory / "metrics.jsonl").open("w", encoding="utf-8")
        self.summary_path = run_directory / "summary.json"

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        self.metrics_file.write(json.dumps(dict(metrics)) + "\n")
        self.metrics_file.flush()

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self.summary_path.write_text(json.dumps(dict(summary), indent=2) + "\n", encoding="utf-8")

    def close(self) -> None:
        self.metrics_file.close()


class TensorBoardLogger(ExperimentLogger):
    """可选的 TensorBoard 后端；仅在配置启用时导入其依赖。"""

    def __init__(self, run_directory: Path) -> None:
        try:
            from torch.utils.tensorboard import SummaryWriter
        except ImportError as error:
            raise ImportError("TensorBoard backend requires `uv add tensorboard`.") from error
        self.writer = SummaryWriter(run_directory / "tensorboard")

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        step = int(metrics["step"])
        for name, value in metrics.items():
            if name not in {"step", "wall_clock_sec"}:
                self.writer.add_scalar(name, value, step)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self.writer.add_text("summary", json.dumps(dict(summary), indent=2))

    def close(self) -> None:
        self.writer.close()


class WandbLogger(ExperimentLogger):
    """可选的 Weights & Biases 后端。"""

    def __init__(self, run_directory: Path, options: Mapping[str, Any], config: Mapping[str, Any]) -> None:
        import wandb

        self.wandb = wandb
        self.run = wandb.init(
            project=options["project"],
            entity=options.get("entity"),
            mode=options.get("mode", "online"),
            dir=str(run_directory),
            config=dict(config),
        )

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        self.wandb.log(dict(metrics), step=int(metrics["step"]))

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        self.run.summary.update(dict(summary))

    def close(self) -> None:
        self.wandb.finish()


class CompositeLogger(ExperimentLogger):
    """向所有已启用的日志后端广播事件。"""

    def __init__(self, loggers: Iterable[ExperimentLogger]) -> None:
        self.loggers = list(loggers)

    def log_metrics(self, metrics: Mapping[str, float | int]) -> None:
        for logger in self.loggers:
            logger.log_metrics(metrics)

    def log_summary(self, summary: Mapping[str, Any]) -> None:
        for logger in self.loggers:
            logger.log_summary(summary)

    def close(self) -> None:
        for logger in self.loggers:
            logger.close()


LoggerFactory = Callable[[Path, Mapping[str, Any], Mapping[str, Any]], ExperimentLogger]
LOGGER_REGISTRY: dict[str, LoggerFactory] = {
    "jsonl": lambda run_directory, _options, _config: JsonlLogger(run_directory),
    "tensorboard": lambda run_directory, _options, _config: TensorBoardLogger(run_directory),
    "wandb": lambda run_directory, options, config: WandbLogger(run_directory, options, config),
}


def build_logger(
    backends: Iterable[Mapping[str, Any]], run_directory: Path, config: Mapping[str, Any]
) -> CompositeLogger:
    """根据配置构造 logger；新增 backend 时只需向 LOGGER_REGISTRY 注册工厂函数。"""
    loggers = []
    for backend in backends:
        options = dict(backend)
        name = str(options.pop("name"))
        try:
            factory = LOGGER_REGISTRY[name]
        except KeyError as error:
            available = ", ".join(sorted(LOGGER_REGISTRY))
            raise ValueError(f"Unknown logger backend {name!r}; available: {available}") from error
        loggers.append(factory(run_directory, options, config))
    return CompositeLogger(loggers)
