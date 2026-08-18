"""Shared byte-level BPE machinery.

All 256 byte values form the initial vocabulary, which guarantees lossless
UTF-8 handling. Subclasses choose how text is split before these operations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from .tokenizer import Tokenizer

Pair = Tuple[int, int]


class BytePairTokenizer(Tokenizer):
    """Base for BPE tokenizers whose initial vocabulary is all 256 bytes."""

    def __init__(self) -> None:
        self.merges: Dict[Pair, int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._reset()

    def _reset(self) -> None:
        # Bytes are the universal starting symbols for every byte-level BPE model.
        self.merges = {}
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    @staticmethod
    def pair_counts(ids: Sequence[int]) -> Counter[Pair]:
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def replace_pair(ids: Sequence[int], pair: Pair, token_id: int) -> List[int]:
        result: List[int] = []
        position = 0
        while position < len(ids):
            if position + 1 < len(ids) and (ids[position], ids[position + 1]) == pair:
                result.append(token_id)
                position += 2
            else:
                result.append(ids[position])
                position += 1
        return result

    def _encode_ids(self, ids: List[int]) -> List[int]:
        while len(ids) > 1:
            pairs = self.pair_counts(ids)
            if not pairs:
                break
            # Lower token IDs were learned first and therefore have higher priority.
            pair = min(pairs, key=lambda candidate: self.merges.get(candidate, float("inf")))
            if pair not in self.merges:
                break
            ids = self.replace_pair(ids, pair, self.merges[pair])
        return ids

    def decode(self, ids: Sequence[int], *, errors: str = "replace") -> str:
        try:
            # Tokens store bytes, not text, so every UTF-8 sequence is reconstructed first.
            data = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return data.decode("utf-8", errors=errors)

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "BytePairTokenizer":
        tokenizer = cls()
        tokenizer._load_merges(model)
        return tokenizer

    def _load_merges(self, model: Dict[str, Any]) -> None:
        raw_merges = model.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("model merges must be a list")

        for token_id, raw_pair in enumerate(raw_merges, start=256):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValueError("every merge must be a two-item list")
            left, right = raw_pair
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError(f"invalid merge: {raw_pair!r}")
            if left not in self.vocab or right not in self.vocab:
                raise ValueError(f"invalid merge: {raw_pair!r}")
            # Each merge may only refer to symbols created by earlier merges.
            self.merges[(left, right)] = token_id
            self.vocab[token_id] = self.vocab[left] + self.vocab[right]

    def _model_data(self) -> Dict[str, Any]:
        return {"merges": [list(pair) for pair in self.merges]}
