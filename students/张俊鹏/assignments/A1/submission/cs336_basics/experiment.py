import json
import os
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class ExperimentTracker:
    """Tracks training experiments with gradient-step and wall-clock logging.

    Writes a JSONL log file with one entry per evaluation step.  Also saves
    the full experiment config so every run is reproducible and comparable.
    """

    name: str
    config: dict[str, Any]
    log_dir: str = "experiments"
    use_wandb: bool = False

    # Internal state
    _start_time: float = field(default_factory=time.time, init=False)
    _history: list[dict] = field(default_factory=list, init=False)
    _log_path: str = field(default="", init=False)
    _wandb_run: Any = field(default=None, init=False)

    def __post_init__(self):
        self._log_dir = os.path.join(self.log_dir, self.name)
        os.makedirs(self._log_dir, exist_ok=True)
        self._log_path = os.path.join(self._log_dir, "metrics.jsonl")
        self._start_time = time.time()

        # Save config for reproducibility
        config_path = os.path.join(self._log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.config, f, indent=2)

        # Weights & Biases (optional)
        if self.use_wandb:
            import wandb
            self._wandb_run = wandb.init(
                project="cs336-assignment1",
                name=self.name,
                config=self.config,
                dir=self._log_dir,
            )

    def log(self, step: int, train_loss: float, val_loss: float, lr: float,
            tokens_processed: int, extra: dict[str, Any] | None = None):
        """Record one evaluation point."""
        entry = {
            "step": step,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "perp": float(np.exp(val_loss)),
            "lr": lr,
            "tokens_processed": tokens_processed,
            "wall_clock_s": time.time() - self._start_time,
        }
        if extra:
            entry.update(extra)

        self._history.append(entry)
        with open(self._log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")

        if self._wandb_run is not None:
            self._wandb_run.log(entry, step=step)

    @property
    def history(self):
        return self._history

    @property
    def best_val_loss(self):
        if not self._history:
            return float("inf")
        return min(e["val_loss"] for e in self._history)

    def summary(self) -> str:
        if not self._history:
            return f"[{self.name}] No evaluations yet."
        best = self.best_val_loss
        final = self._history[-1]
        return (
            f"[{self.name}] best_val_loss={best:.4f}  perp={np.exp(best):.2f}  "
            f"wall_time={final['wall_clock_s']:.0f}s  steps={final['step']}"
        )


def load_experiment(exp_dir: str) -> ExperimentTracker:
    """Load a previously-run experiment from its log directory."""
    config_path = os.path.join(exp_dir, "config.json")
    log_path = os.path.join(exp_dir, "metrics.jsonl")

    with open(config_path) as f:
        config = json.load(f)

    name = os.path.basename(exp_dir)
    tracker = ExperimentTracker(name=name, config=config, log_dir=os.path.dirname(exp_dir))

    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    tracker._history.append(json.loads(line))

    return tracker


def plot_experiments(exp_dirs: list[str], save_path: str | None = None):
    """Plot validation loss curves for multiple experiments."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for d in exp_dirs:
        exp = load_experiment(d)
        steps = [e["step"] for e in exp.history]
        val_loss = [e["val_loss"] for e in exp.history]
        wall = [e["wall_clock_s"] for e in exp.history]

        label = exp.name
        ax1.plot(steps, val_loss, label=label)
        ax2.plot(wall, val_loss, label=label)

    ax1.set_xlabel("Gradient Steps")
    ax1.set_ylabel("Validation Loss")
    ax1.set_title("Validation Loss vs. Steps")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    ax2.set_xlabel("Wall Clock (s)")
    ax2.set_ylabel("Validation Loss")
    ax2.set_title("Validation Loss vs. Wall Time")
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    else:
        plt.show()