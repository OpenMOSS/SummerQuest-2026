import pickle

import regex

from cs336_basics.bpe import PAT

from collections.abc import Iterable, Iterator

class Tokenizer:
    def __init__(
            self,
            vocab:dict[int,bytes],
            merges: list[tuple[bytes,bytes]],
            special_tokens: list[str] | None= None,
    ):
        self.vocab=dict(vocab)
        self.merges=list(merges)
        self.special_tokens=list(special_tokens or [])
        self.token_to_id={
            token_bytes: token_id
            for token_id, token_bytes in self.vocab.items()
        }

        next_token_id=max(self.vocab,default=-1)+1

        self.special_token_to_id: dict[str,int]={}

        for special_token in self.special_tokens:
            special_token_bytes = special_token.encode("utf-8")

            if special_token_bytes not in self.token_to_id:
                self.vocab[next_token_id] = special_token_bytes
                self.token_to_id[special_token_bytes] = next_token_id
                next_token_id += 1

            self.special_token_to_id[special_token] = (
                self.token_to_id[special_token_bytes]
            )

        sorted_special_tokens = sorted(
            self.special_tokens,
            key=len,
            reverse=True,
        )

        if sorted_special_tokens:
            special_pattern="|".join(
                regex.escape(token)
                for token in sorted_special_tokens
            )
            self.special_token_pattern = regex.compile(
                f"({special_pattern})"
            )
        else:
            self.special_token_pattern=None

        self.merge_ranks ={
            pair: rank
            for rank,pair in enumerate(self.merges)
        }

    @classmethod
    def from_files(
        cls,
        vocab_filepath: str,
        merges_filepath: str,
        special_tokens: list[str] | None=None
    ):
        with open(vocab_filepath,"rb") as vocab_file:
            vocab = pickle.load(vocab_file)

        with open(merges_filepath,"rb") as merges_file:
            merges=pickle.load(merges_file)

        return cls(
            vocab = vocab,
            merges=merges,
            special_tokens=special_tokens,
        )


    def _encode_pre_token(
            self,
            pre_token:str,
    ) -> list[int]:
        pre_token_bytes = pre_token.encode("utf-8")
        tokens =[
            bytes([byte_value])
            for byte_value in pre_token_bytes
        ]

        while len(tokens) > 1:
            best_pair: tuple[bytes,bytes] | None=None
            best_rank: int | None=None

            for index in range(len(tokens)-1):
                pair=(
                    tokens[index],
                    tokens[index+1],
                )
                rank = self.merge_ranks.get(pair)

                if rank is not None and(
                    best_rank is None or rank<best_rank
                ):
                    best_pair=pair
                    best_rank = rank

            if best_pair is None:
                break

            merged_tokens: list[bytes] =[]
            index =0

            while index<len(tokens):
                if(
                    index +1 < len(tokens)
                    and tokens[index] ==best_pair[0]
                    and tokens[index+1] ==best_pair[1]
                ):
                    merged_token =(
                        tokens[index] + tokens[index+1]
                    )
                    merged_tokens.append(merged_token)
                    index+=2
                else:
                    merged_tokens.append(tokens[index])
                    index+=1

            tokens=merged_tokens

        return [
            self.token_to_id[token]
            for token in tokens
        ]

    def _encode_ordinary_text(
            self,
            text:str,
    ) -> list[int]:
        token_ids: list[int]=[]

        for match in regex.finditer(PAT,text):
            pre_token=match.group(0)
            pre_token_ids = self._encode_pre_token(pre_token)
            token_ids.extend(pre_token_ids)

        return token_ids

    def encode(
            self,
            text: str,
    )->list[int]:
        if not self.special_tokens:
            return self._encode_ordinary_text(text)

        assert self.special_token_pattern is not None

        token_ids: list[int] = []
        text_segments = self.special_token_pattern.split(text)

        for segment in text_segments:
            if segment == "":
                continue

            if segment in self.special_token_to_id:
                token_ids.append(
                    self.special_token_to_id[segment]
                )
            else:
                token_ids.extend(
                    self._encode_ordinary_text(segment)
                )

        return token_ids

    def encode_iterable(
            self,
            iterable:Iterable[str],
    )-> Iterator[int]:
        for text_chunk in iterable:
            token_ids = self.encode(text_chunk)

            for token_id in token_ids:
                yield token_id

    def decode(
            self,
            ids:list[int],
    ) ->str:
        token_bytes = b"".join(
            self.vocab[token_id]
            for token_id in ids
        )

        return token_bytes.decode(
            "utf-8",
            errors="replace",
        )