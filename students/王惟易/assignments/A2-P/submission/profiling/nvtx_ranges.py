from contextlib import contextmanager

from torch.profiler import record_function

import cs336_basics.model as model_module


@contextmanager
def attention_ranges():
    original_einsum = model_module.einsum
    original_softmax = model_module.softmax

    def profiled_einsum(*args, **kwargs):
        equation = args[-1]
        range_name = None

        if "query d_k" in equation and "key d_k" in equation:
            range_name = "attention/scores"
        elif "query key" in equation and "key d_v" in equation:
            range_name = "attention/value"

        if range_name is None:
            return original_einsum(*args, **kwargs)

        with record_function(range_name):
            return original_einsum(*args, **kwargs)

    def profiled_softmax(x, dim=-1):
        with record_function("attention/softmax"):
            return original_softmax(x, dim=dim)

    model_module.einsum = profiled_einsum
    model_module.softmax = profiled_softmax

    try:
        yield
    finally:
        model_module.einsum = original_einsum
        model_module.softmax = original_softmax
