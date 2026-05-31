import pytest
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

# Preprocessing tests verify the tokenizer handles edge cases your scraper might produce: empty strings, special characters, Singlish
# Production text is messier than training data

@pytest.fixture(scope="module")
def tokenizer():
    """Load tokenizer once for all tests in this module"""
    return AutoTokenizer.from_pretrained(MODEL_NAME)


class TestTokenizer:
    def test_basic_tokenisation(self, tokenizer):
        """Tokenizer should return input_ids and attention_mask"""
        result = tokenizer("This is a test sentence", return_tensors="pt")
        assert "input_ids" in result
        assert "attention_mask" in result

    def test_truncation_at_128(self, tokenizer):
        """Long texts should be truncated to max_length=128"""
        long_text = "word " * 300  # 300 words, well over 128 tokens
        result = tokenizer(
            long_text,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )
        assert result["input_ids"].shape[1] <= 128, \
            f"Expected max 128 tokens, got {result['input_ids'].shape[1]}"

    def test_short_text_not_padded(self, tokenizer):
        """Short texts should not be padded when padding=False"""
        short_text = "good"
        result = tokenizer(short_text, padding=False)
        # Should be short — just a few tokens
        assert len(result["input_ids"]) < 10

    def test_empty_string_handling(self, tokenizer):
        """Tokenizer should handle empty strings without crashing"""
        try:
            result = tokenizer("", truncation=True, max_length=128)
            assert "input_ids" in result
        except Exception as e:
            pytest.fail(f"Tokenizer crashed on empty string: {e}")

    def test_singlish_text(self, tokenizer):
        """Singlish text should tokenize without errors"""
        singlish = "wah lao this queue so long very sian leh"
        result = tokenizer(singlish, truncation=True, max_length=128)
        assert len(result["input_ids"]) > 0

    def test_special_characters(self, tokenizer):
        """Text with special characters should tokenize without crashing"""
        special = "Price is $99.99!!! Buy now 🔥 #singapore"
        try:
            result = tokenizer(special, truncation=True, max_length=128)
            assert len(result["input_ids"]) > 0
        except Exception as e:
            pytest.fail(f"Tokenizer crashed on special characters: {e}")

    def test_batch_tokenisation(self, tokenizer):
        """Batch tokenisation should return consistent shapes"""
        texts = [
            "This is great",
            "Very bad experience",
            "It was okay",
        ]
        result = tokenizer(
            texts,
            truncation=True,
            max_length=128,
            padding=True,
            return_tensors="pt"
        )
        # All sequences should be padded to same length
        assert result["input_ids"].shape[0] == 3
        assert result["attention_mask"].shape[0] == 3