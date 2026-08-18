"""CLI for loading a saved checkpoint and prompting for tokenizer input."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, Type

from pitokenizer import AVAILABLE_MODELS, Tokenizer


def model_class_from_config(path: Path) -> Type[Tokenizer]:
    """Read the model class name from a checkpoint YAML configuration."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ImportError("YAML configuration requires the 'PyYAML' package") from error

    with path.open(encoding="utf-8") as file:
        config: Dict[str, Any] = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("checkpoint configuration must contain a YAML mapping")
    model = config.get("model")
    if not isinstance(model, dict):
        raise ValueError("checkpoint configuration model must be a mapping")
    model_name = model.get("name")
    if not isinstance(model_name, str) or model_name not in AVAILABLE_MODELS:
        raise ValueError(f"unknown checkpoint model: {model_name}")
    return AVAILABLE_MODELS[model_name]


def load_checkpoint(weights_path: Path):
    """Load a tokenizer using the config stored beside its weights file."""
    weights_path = weights_path.resolve()
    if not weights_path.is_file():
        raise FileNotFoundError(f"checkpoint weights do not exist: {weights_path}")

    config_path = weights_path.parent / "config.yaml"
    if not config_path.is_file():
        raise FileNotFoundError(f"checkpoint configuration does not exist: {config_path}")
    return model_class_from_config(config_path).load(weights_path)


def interactive_prompt(tokenizer: Tokenizer) -> None:
    """Encode user input until EOF or an explicit exit command."""
    print("Enter text to tokenize. Type 'exit' or press Ctrl-D to quit.")
    while True:
        try:
            text = input("> ")
        except EOFError:
            print()
            return
        if text.strip().lower() == "exit":
            return

        ids = tokenizer.encode(text)
        print("Input:", text)
        print("Tokens:", ids)
        print("Decoded:", [tokenizer.decode([token_id]) for token_id in ids])
        print("Final Decode:", tokenizer.decode(ids))


def main() -> None:
    parser = argparse.ArgumentParser(description="Open an interactive PiTokenizer session.")
    parser.add_argument("weights", type=Path, help="path to a checkpoint weights.json file")
    args = parser.parse_args()

    interactive_prompt(load_checkpoint(args.weights))


if __name__ == "__main__":
    main()
