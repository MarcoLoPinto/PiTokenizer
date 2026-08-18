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
    """Use OpenAI's fixed cl100k_base vocabulary with configurable special tokens."""

    def __init__(self, special_tokens: Optional[Mapping[str, int]] = None) -> None:
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
        base_encoding = tiktoken.get_encoding("cl100k_base")
        # Reuse exact merge ranks; only the special-token mapping may be extended.
        self._encoding = tiktoken.Encoding(
            name="pitokenizer-gpt4",
            pat_str=GPT4_PATTERN,
            mergeable_ranks=base_encoding._mergeable_ranks,
            special_tokens=self.special_tokens,
        )

    def register_special_tokens(self, special_tokens: Mapping[str, int]) -> None:
        """Add special tokens without changing existing token assignments."""
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
        raise TypeError("GPT4Tokenizer has a fixed cl100k_base vocabulary and cannot be trained")

    def encode(self, text: str, allowed_special: AllowedSpecial = "none") -> List[int]:
        allowed = self._allowed_special_tokens(allowed_special)
        # Treat special-looking text as ordinary text unless the caller opts in.
        return self._encoding.encode(text, allowed_special=allowed, disallowed_special=())

    def decode(self, ids: Sequence[int], *, errors: str = "replace") -> str:
        return self._encoding.decode(list(ids), errors=errors)

    def _model_data(self) -> Dict[str, Any]:
        # Merge ranks are fixed by cl100k_base; special tokens are the mutable state.
        return {"special_tokens": self.special_tokens}

    @classmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "GPT4Tokenizer":
        raw_special_tokens = model.get("special_tokens")
        if not isinstance(raw_special_tokens, dict):
            raise ValueError("model special_tokens must be an object")
        if not all(
            isinstance(token, str) and isinstance(token_id, int)
            for token, token_id in raw_special_tokens.items()
        ):
            raise ValueError("model special_tokens must map strings to integers")
        return cls(special_tokens=raw_special_tokens)
