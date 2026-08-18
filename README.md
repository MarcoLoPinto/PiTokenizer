<p align="center">
  <img src="./assets/logo.png" alt="PiTokenizer" width="256">
</p>

**PiTokenizer** is a small, from-scratch implementation of byte-level Byte Pair Encoding (BPE). It is designed to make the core tokenizer algorithms easy to read, train, save, and reuse.

The project follows an educational approach. GPT-2 and tiktoken are used as behavioural references for regex pre-tokenisation and the fixed GPT-4 encoding.

## Project layout

```text
.
├── configs/                 # YAML training configurations
├── datasets/                # Training text files
├── checkpoints/             # Generated model checkpoints
├── examples/                # Standalone usage examples
├── src/pitokenizer/
│   ├── models/              # Tokenizer implementations
│   ├── train.py             # Training CLI
│   └── inference.py         # Interactive inference CLI
├── tests/
└── pyproject.toml
```

`src/` contains only installable package code. Tests remain at the repository root because they are development code and are not part of the package wheel.

## Installation

The project uses uv. Synchronize the project environment and development dependencies from the repository root:

```bash
uv sync --dev
```

## Tokenizers

| Tokenizer | Purpose | Training |
| --- | --- | --- |
| `BasicTokenizer` | BPE over the full UTF-8 byte stream. Merges can cross word boundaries. | Supported |
| `RegexTokenizer` | BPE applied independently to GPT-style regex pieces. | Supported |
| `GPT4Tokenizer` | Exact fixed `cl100k_base` vocabulary and merge ranks through tiktoken. | Not supported |

All byte-level BPE tokenizers begin with the 256 possible byte values. This makes UTF-8 encoding and decoding lossless.

## Train a tokenizer

Training is driven by YAML. The provided [configs/regex.yaml](configs/regex.yaml) is a complete example.

Supported model names are the keys in `pitokenizer.AVAILABLE_MODELS`: `BasicTokenizer`, `RegexTokenizer`, and `GPT4Tokenizer`. The last one has fixed weights and cannot be trained.

Dataset entries can be absolute paths, paths relative to the configuration file, or file names resolved from `datasets/`.

Run training with either the package entry point:

```bash
uv run pitokenizer-train configs/regex.yaml
```

or the module form:

```bash
uv run python -m pitokenizer.train configs/regex.yaml
```

Each run creates the next available checkpoint directory:

```text
checkpoints/
└── RegexTokenizer/
    └── version_1/
        ├── config.yaml
        └── weights.json
```

`config.yaml` is a copy of the training configuration. `weights.json` contains the model data: merge rules for trainable BPE tokenizers, plus the selected pattern for `RegexTokenizer`.

If the requested vocabulary is larger than the corpus can support, training stops after every possible merge and still saves the resulting valid model. The training CLI reports the vocabulary size that was actually reached.

## Interactive inference

Pass a checkpoint JSON file to the inference CLI:

```bash
uv run pitokenizer-inference checkpoints/RegexTokenizer/version_1/weights.json
```

The CLI reads the sibling `config.yaml` to select the tokenizer class, then prints the encoded IDs and the byte expansion of every individual token. Type `exit` or press Ctrl-D to quit.

## Use a checkpoint from Python

After installing PiTokenizer as a package, load a trainable model explicitly
with its tokenizer class:

```python
from pathlib import Path

from pitokenizer import RegexTokenizer

weights_path = Path("checkpoints/RegexTokenizer/version_1/weights.json")
tokenizer = RegexTokenizer.load(weights_path)

text = "Hello pigeon! Welcome to the world!"
ids = tokenizer.encode(text)

print("Input:", text)
print("Tokens:", ids)
print("Decoded:", [tokenizer.decode([token_id]) for token_id in ids])
print("Final Decode:", tokenizer.decode(ids))
```

Example output:

```text
Input: Hello pigeon! Welcome to the world!
Tokens: [72, 677, 111, 291, 33, 361, 360, 99, 457, 308, 269, 1237, 33]
Decoded: ['H', 'ell', 'o', ' pigeon', '!', ' W', 'el', 'c', 'ome', ' to', ' the', ' world', '!']
Final Decode: Hello pigeon! Welcome to the world!
```

See [examples/load_regex_checkpoint.py](examples/load_regex_checkpoint.py) for the complete executable example.

The saved JSON does not include a tokenizer type field by design. This keeps model files minimal, but means Python code must select the correct concrete class when loading. The inference CLI gets that class from the copied YAML configuration.

## GPT-4-compatible encoding and special tokens

`GPT4Tokenizer` uses tiktoken's published `cl100k_base` merge ranks. It is a fixed inference tokenizer, not a trainable model. Standard special tokens are preserved when saved and loaded.

Special-looking text is treated as ordinary text by default. Explicitly opt in when special tokens should be interpreted:

```python
from pitokenizer import GPT4Tokenizer

tokenizer = GPT4Tokenizer()
ids = tokenizer.encode("<|endoftext|>hello", allowed_special="all")
assert tokenizer.decode(ids) == "<|endoftext|>hello"
```

Use `allowed_special="all"` only for trusted text or when special-token interpretation is intentionally required.

## Testing

Run the full test suite with:

```bash
uv run python -m pytest
```

The tests cover BPE training and encoding, UTF-8 round trips, regex boundaries, GPT-4 special-token handling, persistence, and the model registries.
