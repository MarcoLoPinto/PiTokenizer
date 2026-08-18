"""Common persistence and interface rules shared by tokenizer models.

The base class deliberately keeps no tokenizer identifier in saved files. The
concrete class selected by the caller defines how its own model data is read.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
import json
from pathlib import Path
from typing import Any, Dict, List, Sequence, Union


class Tokenizer(ABC):
    """The common interface for all tokenizer families."""

    @abstractmethod
    def train(self, text: str, vocab_size: int) -> None:
        """Learn a vocabulary from text."""

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text into token IDs."""

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back into text."""

    @abstractmethod
    def _model_data(self) -> Dict[str, Any]:
        """Return the implementation-specific part of a saved model."""

    @classmethod
    @abstractmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "Tokenizer":
        """Build an instance from implementation-specific saved data."""

    def save(self, path: Union[str, Path]) -> None:
        """Save this tokenizer's implementation-specific JSON data."""
        # Subclasses decide the data schema; the base class only owns JSON I/O.
        model = self._model_data()
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(model, file, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Load saved data into the concrete tokenizer class that was called."""
        # Without a type field in the file, loading must be explicit and safe.
        if cls is Tokenizer:
            raise TypeError("call load() on a concrete tokenizer class")
        with Path(path).open(encoding="utf-8") as file:
            model = json.load(file)
        if not isinstance(model, dict):
            raise ValueError("model must be a JSON object")
        return cls._from_model_data(model)
