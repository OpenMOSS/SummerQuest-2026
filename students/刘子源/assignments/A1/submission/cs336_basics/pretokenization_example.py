import os
from typing import BinaryIO


def find_chunk_boundaries(
    file: BinaryIO,
    desired_num_chunks: int,
    split_special_token: bytes,
) -> list[int]:
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    one_chunk = file_size // desired_num_chunks
    cut_where = [i * one_chunk for i in range(desired_num_chunks + 1)]
    cut_where[-1] = file_size
    read_size = 4096  # 4K 一口一口找，别整份文件往内存里塞。

    for cut_i in range(1, len(cut_where) - 1):
        now_pos = cut_where[cut_i]
        file.seek(now_pos)
        while True:
            small_piece = file.read(read_size)
            if small_piece == b"":
                cut_where[cut_i] = file_size
                break
            found_at = small_piece.find(split_special_token)
            if found_at != -1:
                cut_where[cut_i] = now_pos + found_at
                break
            now_pos += read_size

    # 撞到同一个分隔符很正常，去个重就完事了。
    return sorted(set(cut_where))


def iter_pretokenization_chunks(input_path: str, num_processes: int = 4):
    with open(input_path, "rb") as file:
        cut_where = find_chunk_boundaries(file, num_processes, b"<|endoftext|>")
        for start, end in zip(cut_where[:-1], cut_where[1:]):
            file.seek(start)
            yield file.read(end - start).decode("utf-8", errors="ignore")
