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
    """Define the common lifecycle for all tokenizer models.

    Notes
    -----
    Concrete subclasses own their JSON schema. This base class only provides
    safe file I/O and requires callers to choose the concrete class when
    loading, rather than embedding a Python class name in model data.
    """

    @abstractmethod
    def train(self, text: str, vocab_size: int) -> None:
        """Learn tokenizer state from text.

        Parameters
        ----------
        text : str
            Training corpus.
        vocab_size : int
            Requested vocabulary size.
        """

    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Convert text into token IDs.

        Parameters
        ----------
        text : str
            Source text.

        Returns
        -------
        list of int
            Token IDs representing the input.
        """

    @abstractmethod
    def decode(self, ids: Sequence[int]) -> str:
        """Convert token IDs back into text.

        Parameters
        ----------
        ids : sequence of int
            Token IDs to decode.

        Returns
        -------
        str
            Decoded text.
        """

    @abstractmethod
    def _model_data(self) -> Dict[str, Any]:
        """Return JSON-compatible implementation-specific model data.

        Returns
        -------
        dict
            Data required by the concrete tokenizer to rebuild itself.
        """

    @classmethod
    @abstractmethod
    def _from_model_data(cls, model: Dict[str, Any]) -> "Tokenizer":
        """Build an instance from implementation-specific model data.

        Parameters
        ----------
        model : dict
            Parsed JSON data produced by ``_model_data``.

        Returns
        -------
        Tokenizer
            Reconstructed concrete tokenizer.
        """

    def save(self, path: Union[str, Path]) -> None:
        """Save this tokenizer's implementation-specific JSON data.

        Parameters
        ----------
        path : str or pathlib.Path
            Destination JSON file. Parent directories must already exist.
        """
        # Subclasses decide the data schema; the base class only owns JSON I/O.
        model = self._model_data()
        with Path(path).open("w", encoding="utf-8") as file:
            json.dump(model, file, indent=2)
            file.write("\n")

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Tokenizer":
        """Load a JSON model through the concrete class that was called.

        Parameters
        ----------
        path : str or pathlib.Path
            Source JSON model file.

        Returns
        -------
        Tokenizer
            Reconstructed concrete tokenizer.

        Raises
        ------
        TypeError
            If called on the abstract ``Tokenizer`` base class.
        ValueError
            If the JSON root is not an object or the concrete class rejects its
            data.
        """
        # Without a type field in the file, loading must be explicit and safe.
        if cls is Tokenizer:
            raise TypeError("call load() on a concrete tokenizer class")
        with Path(path).open(encoding="utf-8") as file:
            model = json.load(file)
        if not isinstance(model, dict):
            raise ValueError("model must be a JSON object")
        return cls._from_model_data(model)
