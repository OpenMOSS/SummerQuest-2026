"""Print the public, non-secret environment metadata required by A2-K."""

import json

import torch


def main():
    data = {
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        properties = torch.cuda.get_device_properties(0)
        data.update(gpu=properties.name, total_memory_bytes=properties.total_memory)
    print(json.dumps(data, indent=2))


if __name__ == "__main__":
    main()
