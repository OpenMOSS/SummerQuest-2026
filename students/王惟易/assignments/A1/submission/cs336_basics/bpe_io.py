import os
import json
from pathlib import Path
from tempfile import TemporaryDirectory


def save_bpe(
        output_path: str | os.PathLike[str],
        vocab: dict[int, bytes],
        merges: list[tuple[bytes, bytes]]
) -> None:
    data = {
        "vocab": {
            str(token_id): token.hex() for token_id, token in vocab.items()
        },
        "merges": [
            [left.hex(), right.hex()] for left, right in merges
        ]
    }

    with open(output_path, "w") as file:
        json.dump(data, file)


def load_bpe(
        input_path: str | os.PathLike[str],
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:

    with open(input_path) as file:
        data = json.load(file)

    vocab = {
        int(token_id): bytes.fromhex(encoded_token) for token_id, encoded_token in data["vocab"].items()
    }

    merges = [
        (bytes.fromhex(pair[0]), bytes.fromhex(pair[1])) for pair in data["merges"]
    ]

    return vocab, merges


if __name__ == "__main__":
    vocab = {
        0: b"\x00",
        1: b"\xe4",
        2: "你".encode(),
        3: b"<|endoftext|>"
    }

    merges = [
        (b"\xe4", b"\xbd"),
        (b"\xe4\xbd", b"\xa0")
    ]

    with TemporaryDirectory() as temporary_directory:
        output_path = Path(temporary_directory) / "bpe.json"

        save_bpe(output_path, vocab, merges)
        loaded_vocab, loaded_merges = load_bpe(output_path)

        print(loaded_vocab)
        print(loaded_merges)

        assert vocab == loaded_vocab
        assert merges == loaded_merges