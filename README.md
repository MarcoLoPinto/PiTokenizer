<p align="center">
  <img src="./assets/logo.png" alt="PiTokenizer" width="350">
</p>

PiTokenizer is a compact Python library for studying and building tokenizer
implementations. It currently includes byte-level BPE tokenizers, from a small
reference implementation to a GPT-4-compatible fixed encoding.

The code is written for clarity and experimentation. GPT-2 informs the regex
pre-tokenisation behaviour, while tiktoken provides the published ranks for
the fixed `cl100k_base` encoding.

## Repository layout

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

## Quick start

### Install

The project uses uv. From the repository root, synchronize the environment:

```bash
uv sync --dev
```

### Train

Training is configured with YAML. The provided
[configs/regex.yaml](configs/regex.yaml) trains `RegexTokenizer` over the
included dataset:

```yaml
model:
  name: RegexTokenizer
  kwargs:
    pattern: GPT4_PATTERN

training:
  vocab_size: 8192

datasets:
  - wikipedia_pigeons.txt
```

Run it with the installed command:

```bash
uv run pitokenizer-train configs/regex.yaml
```

Each run creates a versioned checkpoint:

```text
checkpoints/RegexTokenizer/version_1/
├── config.yaml
└── weights.json
```

### Inspect a checkpoint

```bash
uv run pitokenizer-inference checkpoints/RegexTokenizer/version_1/weights.json
```

The prompt prints the encoded IDs, the byte expansion of each token, and the
final decoded text. Type `exit` or press Ctrl-D to quit.

## Tokenizers

| Tokenizer | Purpose | Training |
| --- | --- | --- |
| `BasicTokenizer` | BPE over the full UTF-8 byte stream. Merges can cross word boundaries. | Supported |
| `RegexTokenizer` | BPE applied independently to GPT-style regex pieces. | Supported |
| `GPT4Tokenizer` | Exact fixed `cl100k_base` vocabulary and merge ranks through tiktoken. | Not supported |

All byte-level BPE tokenizers begin with the 256 possible byte values. This makes UTF-8 encoding and decoding lossless.

## Training configuration

Supported model names are the keys in `pitokenizer.AVAILABLE_MODELS`:
`BasicTokenizer`, `RegexTokenizer`, and `GPT4Tokenizer`. `GPT4Tokenizer` has
fixed weights and cannot be trained.

Dataset entries can be absolute paths, paths relative to the configuration file, or file names resolved from `datasets/`.

The installed command is:

```bash
uv run pitokenizer-train configs/regex.yaml
```

The module form is also available:

```bash
uv run python -m pitokenizer.train configs/regex.yaml
```

`config.yaml` is a copy of the training configuration. `weights.json` contains the model data: merge rules for trainable BPE tokenizers, plus the selected pattern for `RegexTokenizer`.

If the requested vocabulary is larger than the corpus can support, training
stops after every possible merge and saves the resulting valid model. The CLI
reports the vocabulary size that was actually reached.

## Use a checkpoint from Python

Load a trainable model explicitly with its tokenizer class:

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

> **Note**
> Saved JSON does not include a tokenizer type field. Python code must select
> the concrete class when loading; the inference CLI reads it from the sibling
> `config.yaml`.

## GPT-4-compatible encoding and special tokens

`GPT4Tokenizer` uses tiktoken's published `cl100k_base` merge ranks. It is a
fixed inference tokenizer, not a trainable model. Standard special tokens are
preserved when saved and loaded.

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
