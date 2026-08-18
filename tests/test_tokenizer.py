import json

import pytest

from pitokenizer import (
    CL100K_PATTERN,
    GPT2_PATTERN,
    GPT4_PATTERN,
    GPT4Tokenizer,
    AVAILABLE_MODELS,
    PATTERN_CONSTANTS,
    BasicTokenizer,
    BytePairTokenizer,
    RegexTokenizer,
    Tokenizer,
)


def test_byte_pair_base_builds_the_complete_byte_vocabulary():
    tokenizer = BasicTokenizer()
    assert len(tokenizer.vocab) == 256
    assert tokenizer.vocab[0] == b"\x00"
    assert tokenizer.vocab[255] == b"\xff"


def test_replace_pair_merges_non_overlapping_matches():
    assert BytePairTokenizer.replace_pair([1, 1, 1, 1], (1, 1), 9) == [9, 9]
    assert BytePairTokenizer.replace_pair([1, 1, 1], (1, 1), 9) == [9, 1]


def test_round_trip_unicode():
    tokenizer = BasicTokenizer()
    text = "Hello, world! 日本語 and 😀"
    assert tokenizer.decode(tokenizer.encode(text)) == text


def test_training_matches_the_canonical_toy_example():
    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", 259)
    assert tokenizer.encode("aaabdaaabac") == [258, 100, 258, 97, 99]
    assert tokenizer.decode([258, 100, 258, 97, 99]) == "aaabdaaabac"


def test_training_stops_at_the_largest_vocab_size_the_text_allows():
    tokenizer = BasicTokenizer()
    tokenizer.train("a", 257)
    assert len(tokenizer.vocab) == 256


def test_training_keeps_all_possible_merges_when_the_request_is_too_large():
    tokenizer = BasicTokenizer()
    tokenizer.train("aaaa", 257)
    tokenizer.train("aaaa", 260)

    assert len(tokenizer.vocab) == 258
    assert tokenizer.encode("aaaa") == [257]


def test_save_and_load_rebuild_the_same_model(tmp_path):
    tokenizer = BasicTokenizer()
    tokenizer.train("aaabdaaabac", 259)
    path = tmp_path / "toy.json"
    tokenizer.save(path)
    model = json.loads(path.read_text(encoding="utf-8"))
    assert model == {"merges": [list(pair) for pair in tokenizer.merges]}
    assert "version" not in model
    assert "type" not in model

    restored = BasicTokenizer.load(path)
    assert restored.merges == tokenizer.merges
    assert restored.vocab == tokenizer.vocab
    assert restored.encode("aaabdaaabac") == [258, 100, 258, 97, 99]


def test_base_class_loading_requires_a_concrete_tokenizer(tmp_path):
    path = tmp_path / "model.json"
    path.write_text('{"merges": []}', encoding="utf-8")

    with pytest.raises(TypeError, match="concrete tokenizer"):
        Tokenizer.load(path)


def test_regex_tokenizer_prevents_merges_across_word_boundaries(tmp_path):
    pytest.importorskip("regex")
    tokenizer = RegexTokenizer()
    tokenizer.merges[(ord("o"), ord(" "))] = 256
    tokenizer.vocab[256] = b"o "

    assert tokenizer._pre_tokenize("hello world!") == ["hello", " world", "!"]
    assert 256 not in tokenizer.encode("hello world!")
    assert tokenizer.decode(tokenizer.encode("Hello, world! 日本語 and 😀")) == "Hello, world! 日本語 and 😀"

    path = tmp_path / "regex.json"
    tokenizer.save(path)
    restored = RegexTokenizer.load(path)
    assert restored.pattern == GPT4_PATTERN


def test_gpt4_pattern_is_the_cl100k_pattern():
    assert GPT4_PATTERN == CL100K_PATTERN
    assert GPT4_PATTERN != GPT2_PATTERN


def test_available_models_maps_names_to_tokenizer_classes():
    assert AVAILABLE_MODELS["BasicTokenizer"] is BasicTokenizer
    assert AVAILABLE_MODELS["RegexTokenizer"] is RegexTokenizer
    assert AVAILABLE_MODELS["GPT4Tokenizer"] is GPT4Tokenizer
    assert PATTERN_CONSTANTS["GPT2_PATTERN"] == GPT2_PATTERN
    assert PATTERN_CONSTANTS["CL100K_PATTERN"] == CL100K_PATTERN
    assert PATTERN_CONSTANTS["GPT4_PATTERN"] == GPT4_PATTERN


def test_gpt4_tokenizer_preserves_special_tokens(tmp_path):
    pytest.importorskip("tiktoken")
    tokenizer = GPT4Tokenizer()
    text = "<|endoftext|>hello"

    assert tokenizer.encode("hello") == [15339]
    assert tokenizer.encode(text, allowed_special="all") == [100257, 15339]
    assert 100257 not in tokenizer.encode(text)
    assert tokenizer.decode([100257, 15339]) == text

    path = tmp_path / "gpt4.json"
    tokenizer.save(path)
    restored = GPT4Tokenizer.load(path)
    assert restored.special_tokens == tokenizer.special_tokens
    assert restored.encode(text, allowed_special={"<|endoftext|>"}) == [100257, 15339]
