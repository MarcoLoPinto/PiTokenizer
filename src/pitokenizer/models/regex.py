"""Regex pre-tokenisation for byte-level BPE models.

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
    """Apply byte-level BPE independently to regex-defined text pieces.

    Parameters
    ----------
    pattern : str, default=GPT4_PATTERN
        Unicode-aware regular expression used to split text before BPE.

    Attributes
    ----------
    pattern : str
        Original pattern string saved with the model to preserve token
        boundaries after loading.

    Notes
    -----
    Pair frequencies are aggregated across pieces during training, but a merge
    is only applied inside each individual piece. For example, ``"hello"`` and
    ``" world"`` cannot be merged into a single token.
    """

    def __init__(self, pattern: str = GPT4_PATTERN) -> None:
        """Initialize a tokenizer with a compiled pre-tokenization pattern.

        Parameters
        ----------
        pattern : str, default=GPT4_PATTERN
            Pattern accepted by the third-party ``regex`` package.
        """
        super().__init__()
        # Keep the source string because the selected pattern is part of the model.
        self.pattern = pattern
        self._pattern = regex.compile(pattern)

    def _pre_tokenize(self, text: str) -> List[str]:
        """Split text into the independent BPE inputs.

        Parameters
        ----------
        text : str
            Source text before UTF-8 byte encoding.

        Returns
        -------
        list of str
            Regex matches in input order.
        """
        return self._pattern.findall(text)

    @staticmethod
    def _count_pairs(chunks: Sequence[Sequence[int]]) -> Counter[Pair]:
        """Aggregate pair frequencies without crossing chunk boundaries.

        Parameters
        ----------
        chunks : sequence of sequences of int
            Token IDs for pre-tokenized pieces.

        Returns
        -------
        collections.Counter
            Global pair frequencies used to choose the next merge.
        """
        counts: Counter[Pair] = Counter()
        for chunk in chunks:
            # Count globally for training, but never create a pair across chunks.
            counts.update(BytePairTokenizer.pair_counts(chunk))
        return counts

    def train(self, text: str, vocab_size: int) -> None:
        """Learn BPE merges while preserving regex token boundaries.

        Parameters
        ----------
        text : str
            Training corpus to split with ``pattern`` and encode as UTF-8.
        vocab_size : int
            Requested total vocabulary size including the 256 byte tokens.

        Raises
        ------
        ValueError
            If ``vocab_size`` is smaller than 256.

        Notes
        -----
        Training stops cleanly when all chunks have been reduced as far as
        possible, even if the requested vocabulary size was larger.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        chunks = [list(piece.encode("utf-8")) for piece in self._pre_tokenize(text)]
        # Keep temporary state until the complete train pass has a valid result.
        merges: Dict[Pair, int] = {}
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}
        for token_id in range(256, vocab_size):
            counts = self._count_pairs(chunks)
            if not counts:
                # All pre-tokenized chunks are fully merged; no further rule exists.
                break
            # The most frequent pair receives the next ID and is replaced in every chunk.
            pair = max(counts, key=counts.__getitem__)
            chunks = [self.replace_pair(chunk, pair, token_id) for chunk in chunks]
            merges[pair] = token_id
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        """Encode text without allowing merges across regex pieces.

        Parameters
        ----------
        text : str
            Text to pre-tokenize and encode.

        Returns
        -------
        list of int
            Concatenated token IDs for all encoded pieces.
        """
        ids: List[int] = []
        for piece in self._pre_tokenize(text):
            # Encode each piece independently to preserve the regex boundaries.
            ids.extend(self._encode_ids(list(piece.encode("utf-8"))))
        return ids

    def _model_data(self) -> Dict[str, Any]:
        """Serialize merges together with the boundary-defining regex.

        Returns
        -------
        dict
            JSON-compatible merge data and the original pattern string.
        """
        model = super()._model_data()
        # Loading with a different pattern would change token boundaries and IDs.
        model["pattern"] = self.pattern
        return model

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "RegexTokenizer":
        """Rebuild a regex tokenizer from serialized pattern and merges.

        Parameters
        ----------
        model : dict
            JSON model data containing ``pattern`` and ordered ``merges``.

        Returns
        -------
        RegexTokenizer
            Tokenizer configured with the saved pattern and merge rules.

        Raises
        ------
        ValueError
            If the saved pattern is absent or not a string.
        """
        pattern = model.get("pattern")
        if not isinstance(pattern, str):
            raise ValueError("model pattern must be a string")
        tokenizer = cls(pattern=pattern)
        tokenizer._load_merges(model)
        return tokenizer
