from __future__ import annotations

import json
import platform

import torch


def main() -> None:
    data = {
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        data.update(
            gpu_name=torch.cuda.get_device_name(0), cuda_runtime=torch.version.cuda
        )
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
