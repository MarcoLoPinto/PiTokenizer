"""Load and use a trained RegexTokenizer from another Python project."""

from pathlib import Path

from pitokenizer import RegexTokenizer


WEIGHTS_PATH = Path("checkpoints/RegexTokenizer/version_1/weights.json")

def main() -> None:
    tokenizer = RegexTokenizer.load(WEIGHTS_PATH)
    text = "Hello pigeon! Welcome to the world!"
    ids = tokenizer.encode(text)

    print("Input:", text)
    print("Tokens:", ids)
    print("Decoded:", [tokenizer.decode([token_id]) for token_id in ids])
    print("Final Decode:", tokenizer.decode(ids))

if __name__ == "__main__":
    main()
