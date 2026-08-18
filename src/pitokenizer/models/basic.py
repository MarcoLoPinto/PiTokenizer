"""The smallest useful byte-level BPE tokenizer model.

This educational baseline applies BPE to the complete UTF-8 byte stream. It
intentionally has no pre-tokenisation, so merges may cross word boundaries.
"""

from __future__ import annotations

from typing import Dict, List

from .bpe import BytePairTokenizer, Pair


class BasicTokenizer(BytePairTokenizer):
    """Train byte-level BPE over one complete UTF-8 byte stream.

    No regex or whitespace pre-tokenisation is performed. Consequently, a
    learned merge may include a space or span adjacent words. This makes the
    class a compact reference implementation of the BPE algorithm.

    Notes
    -----
    Training starts with token IDs 0 through 255, one for every byte value.
    New IDs are assigned in merge-learning order, beginning at 256.
    """

    def train(self, text: str, vocab_size: int) -> None:
        """Learn BPE merge rules from text.

        Parameters
        ----------
        text : str
            Training corpus. It is encoded as UTF-8 before pair statistics are
            collected.
        vocab_size : int
            Requested total vocabulary size, including the 256 initial byte
            tokens.

        Raises
        ------
        ValueError
            If ``vocab_size`` is smaller than 256.

        Notes
        -----
        If the corpus no longer contains an adjacent pair before the requested
        size is reached, training stops and retains the largest valid model.
        """
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
                # The corpus is fully merged, so publish the smaller valid model.
                break

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
        """Encode text using the learned BPE merge priority.

        Parameters
        ----------
        text : str
            Text to encode as UTF-8 bytes.

        Returns
        -------
        list of int
            Token IDs after all applicable merges have been applied.
        """
        return self._encode_ids(list(text.encode("utf-8")))
