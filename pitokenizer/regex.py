"""Regex pre-tokenisation for byte-level BPE.

The regex divides text into GPT-style pieces before BPE runs. This prevents a
merge from spanning a word, number, punctuation, or whitespace boundary.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence

import regex

from .bpe import BytePairTokenizer, Pair


GPT2_PATTERN = r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
CL100K_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}++|\p{N}{1,3}+| ?[^\s\p{L}\p{N}]++[\r\n]*+|\s++$|\s*[\r\n]|\s+(?!\S)|\s"""
GPT4_PATTERN = CL100K_PATTERN


class RegexTokenizer(BytePairTokenizer):
    """Byte-level BPE that applies merges independently to GPT-style pieces."""

    def __init__(self, pattern: str = GPT4_PATTERN) -> None:
        super().__init__()
        # Keep the source string because the selected pattern is part of the model.
        self.pattern = pattern
        self._pattern = regex.compile(pattern)

    def _pre_tokenize(self, text: str) -> List[str]:
        return self._pattern.findall(text)

    @staticmethod
    def _count_pairs(chunks: Sequence[Sequence[int]]) -> Counter[Pair]:
        counts: Counter[Pair] = Counter()
        for chunk in chunks:
            # Count globally for training, but never create a pair across chunks.
            counts.update(BytePairTokenizer.pair_counts(chunk))
        return counts

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        chunks = [list(piece.encode("utf-8")) for piece in self._pre_tokenize(text)]
        merges: Dict[Pair, int] = {}
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for token_id in range(256, vocab_size):
            counts = self._count_pairs(chunks)
            if not counts:
                # All pre-tokenized chunks are fully merged; no further rule exists.
                break
            pair = max(counts, key=counts.__getitem__)
            chunks = [self.replace_pair(chunk, pair, token_id) for chunk in chunks]
            merges[pair] = token_id
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        ids: List[int] = []
        for piece in self._pre_tokenize(text):
            # Encode each piece independently to preserve the regex boundaries.
            ids.extend(self._encode_ids(list(piece.encode("utf-8"))))
        return ids

    def _model_data(self) -> Dict[str, Any]:
        model = super()._model_data()
        # Loading with a different pattern would change token boundaries and IDs.
        model["pattern"] = self.pattern
        return model

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "RegexTokenizer":
        pattern = model.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("model pattern must be a string")
        tokenizer = cls(pattern=pattern)
        tokenizer._load_merges(model)
        return tokenizer
