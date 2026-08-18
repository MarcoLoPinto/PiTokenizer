"""Train a tokenizer from YAML and create a versioned checkpoint."""

from __future__ import annotations

import argparse
from pathlib import Path
import re
import shutil
from typing import Any, Dict, List, Mapping

from pitokenizer import AVAILABLE_MODELS, PATTERN_CONSTANTS, Tokenizer
from pitokenizer.models.gpt4 import GPT4Tokenizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
VERSION_PATTERN = re.compile(r"version_(\d+)$")


def load_yaml_config(path: Path) -> Dict[str, Any]:
    """Read and validate a YAML configuration file."""
    try:
        import yaml
    except ModuleNotFoundError as error:
        raise ImportError("YAML configuration requires the 'PyYAML' package") from error

    with path.open(encoding="utf-8") as file:
        config = yaml.safe_load(file)
    if not isinstance(config, dict):
        raise ValueError("configuration must contain a YAML mapping")
    return config


def tokenizer_from_config(config: Mapping[str, Any]):
    """Construct one supported tokenizer from the model configuration."""
    model = config.get("model")
    if not isinstance(model, Mapping):
        raise ValueError("configuration model must be a mapping")

    tokenizer_name = model.get("name")
    if not isinstance(tokenizer_name, str) or tokenizer_name not in AVAILABLE_MODELS:
        available = ", ".join(sorted(AVAILABLE_MODELS))
        raise ValueError(f"unknown tokenizer name; choose one of: {available}")

    kwargs = model.get("kwargs", {})
    if not isinstance(kwargs, Mapping):
        raise ValueError("configuration model kwargs must be a mapping")
    resolved_kwargs = dict(kwargs)

    # YAML stores symbolic pattern names as strings; resolve only the known names.
    pattern = resolved_kwargs.get("pattern")
    if isinstance(pattern, str) and pattern in PATTERN_CONSTANTS:
        resolved_kwargs["pattern"] = PATTERN_CONSTANTS[pattern]

    return AVAILABLE_MODELS[tokenizer_name](**resolved_kwargs)


def dataset_paths(config: Mapping[str, Any], config_path: Path) -> List[Path]:
    """Resolve dataset paths from the YAML directory or project datasets folder."""
    datasets = config.get("datasets")
    if not isinstance(datasets, list) or not datasets:
        raise ValueError("configuration datasets must be a non-empty list")

    paths: List[Path] = []
    for dataset in datasets:
        if not isinstance(dataset, str) or not dataset:
            raise ValueError("every dataset entry must be a non-empty string")
        requested_path = Path(dataset)
        path = config_path.parent / requested_path
        if not path.is_file():
            path = PROJECT_ROOT / "datasets" / requested_path
        if not path.is_file():
            raise FileNotFoundError(f"dataset does not exist: {dataset}")
        paths.append(path)
    return paths


def training_vocab_size(config: Mapping[str, Any]) -> int:
    """Return the required target size for the trainable BPE vocabulary."""
    training = config.get("training")
    if not isinstance(training, Mapping):
        raise ValueError("configuration training must be a mapping")
    vocab_size = training.get("vocab_size")
    if not isinstance(vocab_size, int):
        raise ValueError("configuration training vocab_size must be an integer")
    return vocab_size


def next_checkpoint_directory(model_name: str) -> Path:
    """Choose the next monotonic version directory for a model name."""
    model_directory = PROJECT_ROOT / "checkpoints" / model_name
    model_directory.mkdir(parents=True, exist_ok=True)
    versions = []
    for path in model_directory.iterdir():
        match = VERSION_PATTERN.fullmatch(path.name)
        if path.is_dir() and match:
            versions.append(int(match.group(1)))
    return model_directory / f"version_{max(versions, default=0) + 1}"


def train_from_config(config_path: Path) -> Path:
    """Train the configured tokenizer and return its checkpoint directory."""
    config_path = config_path.resolve()
    config = load_yaml_config(config_path)
    tokenizer: Tokenizer = tokenizer_from_config(config)
    if isinstance(tokenizer, GPT4Tokenizer):
        raise ValueError("GPT4Tokenizer has fixed weights and cannot be trained")

    model = config["model"]
    model_name = model["name"]
    text = "\n".join(path.read_text(encoding="utf-8") for path in dataset_paths(config, config_path))
    requested_vocab_size = training_vocab_size(config)
    tokenizer.train(text, requested_vocab_size)

    checkpoint_directory = next_checkpoint_directory(model_name)
    checkpoint_directory.mkdir()
    tokenizer.save(checkpoint_directory / "weights.json")
    # A fixed name lets inference recover the construction settings from weights alone.
    shutil.copy2(config_path, checkpoint_directory / "config.yaml")
    actual_vocab_size = len(tokenizer.vocab)
    if actual_vocab_size < requested_vocab_size:
        print(
            f"Requested vocab_size {requested_vocab_size}; "
            f"saved the maximum reachable size {actual_vocab_size}."
        )
    return checkpoint_directory


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PiTokenizer model from YAML.")
    parser.add_argument("config", type=Path, help="path to the YAML configuration file")
    args = parser.parse_args()

    checkpoint_directory = train_from_config(args.config)
    print(f"Checkpoint saved to {checkpoint_directory}")


if __name__ == "__main__":
    main()
