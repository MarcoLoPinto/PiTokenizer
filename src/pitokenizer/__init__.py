"""PiTokenizer: small, from-scratch tokenizer implementations."""

from .models.basic import BasicTokenizer
from .models.bpe import BytePairTokenizer
from .models.gpt4 import GPT4Tokenizer
from .models.regex import CL100K_PATTERN, GPT2_PATTERN, GPT4_PATTERN, RegexTokenizer
from .models.tokenizer import Tokenizer

# Central registry used by configuration-driven model loading.
AVAILABLE_MODELS = {
    "BasicTokenizer": BasicTokenizer,
    "RegexTokenizer": RegexTokenizer,
    "GPT4Tokenizer": GPT4Tokenizer,
}

# Named regex patterns accepted by YAML model configurations.
PATTERN_CONSTANTS = {
    "GPT2_PATTERN": GPT2_PATTERN,
    "CL100K_PATTERN": CL100K_PATTERN,
    "GPT4_PATTERN": GPT4_PATTERN,
}

__all__ = [
    "Tokenizer",
    "BytePairTokenizer",
    "BasicTokenizer",
    "RegexTokenizer",
    "GPT4Tokenizer",
    "AVAILABLE_MODELS",
    "PATTERN_CONSTANTS",
    "GPT2_PATTERN",
    "CL100K_PATTERN",
    "GPT4_PATTERN",
]
