# A1 实验配置

每个正式 JSON 文件只描述一个训练 run。先用 `--dry-run` 检查，再去掉该参数启动。

本地 3-step smoke 配置位于已排除的 `.local_checks/configs/`，不会进入最终提交：

```powershell
uv run python scripts/run_experiment.py --config .local_checks/configs/smoke_baseline.json --dry-run
uv run python scripts/run_experiment.py --config .local_checks/configs/smoke_baseline.json
```

这些 smoke 配置只验证代码路径，使用 5MB fixture 和 3 个 step，不能作为正式实验结果。

正式 TinyStories/OWT 配置将在完整数据、训练设备和 token 预算确认后创建。公开配置只使用相对
路径，不记录学校云主机、账号、IP、内部目录或凭据。

GPU benchmark 配置还可以显式记录：

- `matmul_precision`: `highest`、`high` 或 `medium`；
- `compile_mode`: `none`、`default` 或 `reduce-overhead`；
- `amp`: `none` 或 `bf16`；
- `fail_on_non_finite`: 遇到非有限 loss/gradient 时是否立即写出失败 summary 并终止。

正式实验必须使用唯一 `output_dir`，不能让两个 Slurm task 写入同一目录。
