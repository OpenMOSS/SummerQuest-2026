import torch


def main() -> None:
    # Case 1:
    # FP32 accumulator + FP32 input
    print("===== case_1_fp32_accumulator_fp32_input =====")

    s = torch.tensor(
        0,
        dtype=torch.float32,
    )

    for i in range(1000):
        s += torch.tensor(
            0.01,
            dtype=torch.float32,
        )

    print("value:", s)
    print("dtype:", s.dtype)
    print("as_float:", float(s))

    # Case 2:
    # FP16 accumulator + FP16 input
    print("===== case_2_fp16_accumulator_fp16_input =====")

    s = torch.tensor(
        0,
        dtype=torch.float16,
    )

    for i in range(1000):
        s += torch.tensor(
            0.01,
            dtype=torch.float16,
        )

    print("value:", s)
    print("dtype:", s.dtype)
    print("as_float:", float(s))

    # Case 3:
    # FP32 accumulator + FP16 input
    print("===== case_3_fp32_accumulator_fp16_input =====")

    s = torch.tensor(
        0,
        dtype=torch.float32,
    )

    for i in range(1000):
        s += torch.tensor(
            0.01,
            dtype=torch.float16,
        )

    print("value:", s)
    print("dtype:", s.dtype)
    print("as_float:", float(s))

    # Case 4:
    # FP32 accumulator + FP16 input explicitly converted to FP32
    print("===== case_4_fp32_accumulator_explicit_cast =====")

    s = torch.tensor(
        0,
        dtype=torch.float32,
    )

    for i in range(1000):
        x = torch.tensor(
            0.01,
            dtype=torch.float16,
        )
        s += x.type(torch.float32)

    print("value:", s)
    print("dtype:", s.dtype)
    print("as_float:", float(s))


if __name__ == "__main__":
    main()