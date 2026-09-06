import json
import platform

PUBLIC_KEYS = ["python", "torch", "torch_cuda", "cuda_available", "gpus"]


def public_env() -> dict:
    env = {
        "python": platform.python_version(),
        "torch": None,
        "torch_cuda": None,
        "cuda_available": False,
        "gpus": [],
    }
    try:
        import torch

        env["torch"] = getattr(torch, "__version__", None)
        cuda_ver = getattr(torch.version, "cuda", None)
        env["torch_cuda"] = cuda_ver
        avail = bool(torch.cuda.is_available())
        env["cuda_available"] = avail
        if avail:
            for i in range(torch.cuda.device_count()):
                try:
                    props = torch.cuda.get_device_properties(i)
                    env["gpus"].append(
                        {
                            "name": torch.cuda.get_device_name(i),
                            "total_memory_bytes": int(props.total_memory),
                            "capability": f"{props.major}.{props.minor}",
                        }
                    )
                except Exception:
                    continue
    except Exception as exc:  # import itself may fail on a CPU-only box
        env["torch_error"] = f"{type(exc).__name__}: {exc}"
    return env


def public_env_json() -> str:
    return json.dumps(public_env(), indent=2, default=str)
