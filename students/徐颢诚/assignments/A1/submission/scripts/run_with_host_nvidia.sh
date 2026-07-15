#!/usr/bin/env bash
# 让当前进程加载与宿主机内核 NVIDIA 模块相同版本的 libcuda/NVML。
set -euo pipefail

driver_version=$(sed -n 's/.*Kernel Module  \([0-9.]*\).*/\1/p' /proc/driver/nvidia/version | head -n 1)
library_directory=/usr/lib/x86_64-linux-gnu
nvml_library="$library_directory/libnvidia-ml.so.$driver_version"
cuda_library="$library_directory/libcuda.so.$driver_version"

if [[ -z "$driver_version" || ! -f "$nvml_library" || ! -f "$cuda_library" ]]; then
    echo "无法找到与内核 NVIDIA 驱动匹配的用户态库（version: ${driver_version:-unknown}）。" >&2
    exit 1
fi

export LD_PRELOAD="$nvml_library:$cuda_library${LD_PRELOAD:+:$LD_PRELOAD}"
# Miniconda 自带的 ncurses 不一定能发现系统 terminfo；nvitop 的 curses UI 需要它。
if [[ -d /lib/terminfo ]]; then
    export TERMINFO=/lib/terminfo
fi
exec uv run "$@"
