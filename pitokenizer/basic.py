"""The smallest useful byte-level BPE tokenizer.

This educational baseline applies BPE to the complete UTF-8 byte stream. It
intentionally has no pre-tokenisation, so merges may cross word boundaries.
"""

from __future__ import annotations

from typing import Dict, List

from .bpe import BytePairTokenizer, Pair


class BasicTokenizer(BytePairTokenizer):
    """BPE over the entire UTF-8 byte stream, without pre-tokenisation."""

    def train(self, text: str, vocab_size: int) -> None:
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256")

        ids = list(text.encode("utf-8"))
        # Build locally so an impossible request cannot leave a partial model behind.
        merges: Dict[Pair, int] = {}
        vocab = {token_id: bytes([token_id]) for token_id in range(256)}

        # IDs 0 through 255 are the original byte vocabulary. Each iteration learns
        # one new merge and assigns it the next consecutive token ID.
        for token_id in range(256, vocab_size):
            counts = self.pair_counts(ids)
            if not counts:
                raise ValueError(
                    f"vocab_size {vocab_size} exceeds the maximum "
                    f"{token_id} for this training text"
                )

            # Select the most frequent adjacent pair in the current representation.
            # If frequencies are equal, Counter preserves first-seen order and max()
            # keeps that first pair, making the training result deterministic.
            pair = max(counts, key=counts.__getitem__)

            # Replacing every non-overlapping match is the BPE training operation.
            # For example, [97, 97, 97, 97] becomes [token_id, token_id] when
            # the selected pair is (97, 97).
            ids = self.replace_pair(ids, pair, token_id)

            # Store both the rule used during encoding and its byte expansion used
            # during decoding. A merge can reference earlier learned merge tokens.
            merges[pair] = token_id
            vocab[token_id] = vocab[pair[0]] + vocab[pair[1]]

        # Publish the fully trained state only after every requested merge succeeded.
        self.merges = merges
        self.vocab = vocab

    def encode(self, text: str) -> List[int]:
        return self._encode_ids(list(text.encode("utf-8")))
