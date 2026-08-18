"""Shared byte-level BPE machinery for tokenizer models.

All 256 byte values form the initial vocabulary, which guarantees lossless
UTF-8 handling. Subclasses choose how text is split before these operations.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Sequence, Tuple

from .tokenizer import Tokenizer

Pair = Tuple[int, int]


class BytePairTokenizer(Tokenizer):
    """Provide the shared state and operations of byte-level BPE.

    Attributes
    ----------
    merges : dict[tuple[int, int], int]
        Mapping from a merged pair to its assigned token ID. Smaller assigned
        IDs have higher encoding priority because they were learned first.
    vocab : dict[int, bytes]
        Byte expansion for every token ID. It makes decoding independent from
        the original training text.

    Notes
    -----
    Subclasses decide whether BPE runs over the full stream or separately over
    pre-tokenized chunks. This class never interprets bytes as text while
    merging, preserving arbitrary UTF-8 input.
    """

    def __init__(self) -> None:
        """Initialize the byte vocabulary and an empty merge table."""
        self.merges: Dict[Pair, int] = {}
        self.vocab: Dict[int, bytes] = {}
        self._reset()

    def _reset(self) -> None:
        """Restore the initial 256-byte vocabulary.

        Notes
        -----
        This is used before training so token IDs are always allocated from the
        same deterministic base state.
        """
        # Bytes are the universal starting symbols for every byte-level BPE model.
        self.merges = {}
        self.vocab = {token_id: bytes([token_id]) for token_id in range(256)}

    @staticmethod
    def pair_counts(ids: Sequence[int]) -> Counter[Pair]:
        """Count adjacent token pairs.

        Parameters
        ----------
        ids : sequence of int
            Current token representation of one byte stream or one pre-token.

        Returns
        -------
        collections.Counter
            Frequency for each adjacent pair. Overlapping occurrences count
            here; replacement later uses only non-overlapping occurrences.
        """
        return Counter(zip(ids, ids[1:]))

    @staticmethod
    def replace_pair(ids: Sequence[int], pair: Pair, token_id: int) -> List[int]:
        """Replace non-overlapping occurrences of one pair.

        Parameters
        ----------
        ids : sequence of int
            Current token IDs.
        pair : tuple of int
            Adjacent IDs selected for one BPE merge.
        token_id : int
            New ID representing ``pair``.

        Returns
        -------
        list of int
            IDs after the replacement pass.
        """
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
        """Apply learned merges to one independent sequence of token IDs.

        Parameters
        ----------
        ids : list of int
            Initial byte IDs for text or one pre-tokenized chunk.

        Returns
        -------
        list of int
            Representation produced by repeatedly applying available merges in
            learned priority order.
        """
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
        """Decode token IDs back to text.

        Parameters
        ----------
        ids : sequence of int
            Token IDs to expand through ``vocab``.
        errors : str, default="replace"
            UTF-8 decoding error policy.

        Returns
        -------
        str
            Decoded UTF-8 text.

        Raises
        ------
        ValueError
            If an ID does not belong to the vocabulary.
        """
        try:
            # Tokens store bytes, not text, so every UTF-8 sequence is reconstructed first.
            data = b"".join(self.vocab[token_id] for token_id in ids)
        except KeyError as error:
            raise ValueError(f"unknown token id: {error.args[0]}") from error
        return data.decode("utf-8", errors=errors)

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "BytePairTokenizer":
        """Rebuild a byte-level BPE tokenizer from serialized merges.

        Parameters
        ----------
        model : dict
            Model data containing ordered merge pairs.

        Returns
        -------
        BytePairTokenizer
            A tokenizer with its byte vocabulary reconstructed.
        """
        tokenizer = cls()
        tokenizer._load_merges(model)
        return tokenizer

    def _load_merges(self, model: Dict[str, Any]) -> None:
        """Validate ordered merges and rebuild their byte expansions.

        Parameters
        ----------
        model : dict
            Serialized model data containing a ``merges`` list.

        Raises
        ------
        ValueError
            If the merge list is malformed or references a token unavailable at
            that point in the merge order.
        """
        raw_merges = model.get("merges")
        if not isinstance(raw_merges, list):
            raise ValueError("model merges must be a list")

        for token_id, raw_pair in enumerate(raw_merges, start=256):
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValueError("every merge must be a two-item list")
            left, right = raw_pair
            if not isinstance(left, int) or not isinstance(right, int):
                raise ValueError(f"invalid merge: {raw_pair!r}")
            missing_ids = [token for token in (left, right) if token not in self.vocab]
            if missing_ids:
                raise ValueError(
                    f"merge token {token_id} references unavailable token IDs: {missing_ids}"
                )
            pair = (left, right)
            if pair in self.merges:
                raise ValueError(f"duplicate merge pair: {pair}")
            # Each merge may only refer to symbols created by earlier merges.
            self.merges[pair] = token_id
            self.vocab[token_id] = self.vocab[left] + self.vocab[right]

    def _model_data(self) -> Dict[str, Any]:
        """Serialize merge rules in token-ID creation order.

        Returns
        -------
        dict
            JSON-compatible model data. Byte expansions are derived on load and
            therefore do not need to be stored.
        """
        return {"merges": [list(pair) for pair in self.merges]}
