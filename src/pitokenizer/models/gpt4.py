"""A fixed GPT-4-compatible tokenizer model backed by tiktoken's cl100k_base.

Unlike the trainable tokenizers, this class reuses the published GPT-4 merge
ranks. It also preserves special token assignments when a model is saved.
"""

from __future__ import annotations

from collections.abc import Collection, Mapping
from typing import Any, Dict, List, Optional, Sequence, Set, Union

try:
    import tiktoken
except ModuleNotFoundError:
    tiktoken = None

from .regex import GPT4_PATTERN
from .tokenizer import Tokenizer


AllowedSpecial = Union[str, Collection[str]]


class GPT4Tokenizer(Tokenizer):
    """Use the fixed ``cl100k_base`` encoding with preserved special tokens.

    Parameters
    ----------
    special_tokens : mapping of str to int, optional
        Additional special-token assignments. Standard ``cl100k_base`` special
        tokens are always retained.

    Attributes
    ----------
    special_tokens : dict[str, int]
        Complete mapping of standard and registered special-token strings to
        their IDs.

    Notes
    -----
    This class intentionally does not train. Its purpose is to reproduce the
    published GPT-4-compatible ranks, not to learn replacement ranks from a
    local corpus.
    """

    def __init__(self, special_tokens: Optional[Mapping[str, int]] = None) -> None:
        """Build an encoding from tiktoken's published ``cl100k_base`` data.

        Parameters
        ----------
        special_tokens : mapping of str to int, optional
            Extra special tokens to register after loading the standard mapping.

        Raises
        ------
        ImportError
            If the optional runtime dependency ``tiktoken`` is unavailable.
        """
        if tiktoken is None:
            raise ImportError("GPT4Tokenizer requires the 'tiktoken' package")

        base_encoding = tiktoken.get_encoding("cl100k_base")
        # Start from the published assignments so standard GPT-4 markers remain valid.
        self.special_tokens = dict(base_encoding._special_tokens)
        if special_tokens is not None:
            self.register_special_tokens(special_tokens)
        else:
            self._build_encoding()

    def _build_encoding(self) -> None:
        """Create a tiktoken encoding from fixed ranks and current specials.

        Notes
        -----
        Rebuilding is required after special tokens change because tiktoken
        stores their recognition rules inside the encoding instance.
        """
        base_encoding = tiktoken.get_encoding("cl100k_base")
        # Reuse exact merge ranks; only the special-token mapping may be extended.
        self._encoding = tiktoken.Encoding(
            name="pitokenizer-gpt4",
            pat_str=GPT4_PATTERN,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def register_special_tokens(self, special_tokens: Mapping[str, int]) -> None:
        """Register additional special tokens without changing existing IDs.

        Parameters
        ----------
        special_tokens : mapping of str to int
            Names and IDs to add. IDs must not collide with mergeable tokens or
            existing special-token assignments.

        Raises
        ------
        ValueError
            If a name or ID is invalid, or if an assignment conflicts with an
            existing token.
        """
        if not special_tokens:
            self._build_encoding()
            return

        updated_tokens = dict(self.special_tokens)
        used_ids = {token_id: token for token, token_id in updated_tokens.items()}
        base_encoding = tiktoken.get_encoding("cl100k_base")
        maximum_mergeable_id = max(base_encoding._mergeable_ranks.values())
        for token, token_id in special_tokens.items():
            if not isinstance(token, str) or not token:
                raise ValueError("special token names must be non-empty strings")
            if not isinstance(token_id, int) or token_id <= maximum_mergeable_id:
                raise ValueError("special token IDs must be greater than 100255")
            existing_id = updated_tokens.get(token)
            if existing_id is not None and existing_id != token_id:
                raise ValueError(f"special token already has ID {existing_id}: {token}")
            existing_token = used_ids.get(token_id)
            if existing_token is not None and existing_token != token:
                raise ValueError(f"special token ID already belongs to {existing_token}")
            updated_tokens[token] = token_id
            used_ids[token_id] = token

        self.special_tokens = updated_tokens
        self._build_encoding()

    def _allowed_special_tokens(self, allowed_special: AllowedSpecial) -> Set[str]:
        """Normalize the caller's special-token permission setting.

        Parameters
        ----------
        allowed_special : {"all", "none"} or collection of str
            Special-token names allowed to be recognized during encoding.

        Returns
        -------
        set of str
            Permitted special-token names.

        Raises
        ------
        ValueError
            If a string other than ``"all"`` or ``"none"`` is supplied, or if
            a requested token is unknown.
        """
        if allowed_special == "all":
            return set(self.special_tokens)
        if allowed_special == "none":
            return set()
        if isinstance(allowed_special, str):
            raise ValueError("allowed_special must be 'all', 'none', or a collection of tokens")

        allowed = set(allowed_special)
        unknown_tokens = allowed.difference(self.special_tokens)
        if unknown_tokens:
            unknown = ", ".join(sorted(unknown_tokens))
            raise ValueError(f"unknown special tokens: {unknown}")
        return allowed

    def train(self, text: str, vocab_size: int) -> None:
        """Reject training because this tokenizer has fixed published weights.

        Parameters
        ----------
        text : str
            Unused training text.
        vocab_size : int
            Unused requested vocabulary size.

        Raises
        ------
        TypeError
            Always, because training would no longer reproduce ``cl100k_base``.
        """
        raise TypeError("GPT4Tokenizer has a fixed cl100k_base vocabulary and cannot be trained")

    def encode(self, text: str, allowed_special: AllowedSpecial = "none") -> List[int]:
        """Encode text with optional explicit special-token recognition.

        Parameters
        ----------
        text : str
            Text to encode.
        allowed_special : {"all", "none"} or collection of str, default="none"
            Which special tokens may be interpreted. ``"none"`` treats text
            that resembles a special token as ordinary text.

        Returns
        -------
        list of int
            Token IDs produced by ``cl100k_base``.
        """
        allowed = self._allowed_special_tokens(allowed_special)
        # Treat special-looking text as ordinary text unless the caller opts in.
        return self._encoding.encode(text, allowed_special=allowed, disallowed_special=())

    def decode(self, ids: Sequence[int], *, errors: str = "replace") -> str:
        """Decode GPT-4-compatible IDs back to text.

        Parameters
        ----------
        ids : sequence of int
            IDs produced by this tokenizer.
        errors : str, default="replace"
            UTF-8 decoding error policy forwarded to tiktoken.

        Returns
        -------
        str
            Decoded text, including special-token strings when present.
        """
        return self._encoding.decode(list(ids), errors=errors)

    def _model_data(self) -> Dict[str, Any]:
        """Serialize the only mutable GPT-4 tokenizer state.

        Returns
        -------
        dict
            Complete special-token mapping. Merge ranks are rebuilt from the
            fixed ``cl100k_base`` encoding.
        """
        # Merge ranks are fixed by cl100k_base; special tokens are the mutable state.
        return {"special_tokens": self.special_tokens}

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "GPT4Tokenizer":
        """Restore special-token assignments over fixed GPT-4 merge ranks.

        Parameters
        ----------
        model : dict
            JSON model data containing a ``special_tokens`` mapping.

        Returns
        -------
        GPT4Tokenizer
            Tokenizer with the saved special-token assignments.

        Raises
        ------
        ValueError
            If ``special_tokens`` is missing or does not map strings to IDs.
        """
        raw_special_tokens = model.get("special_tokens")
        if not isinstance(raw_special_tokens, dict):
            raise ValueError("model special_tokens must be an object")
        if not all(
            isinstance(token, str) and isinstance(token_id, int)
            for token, token_id in raw_special_tokens.items()
        ):
            raise ValueError("model special_tokens must map strings to integers")
        return cls(special_tokens=raw_special_tokens)
