import pytest
import pandas as pd
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

# === Tests for scrape_huggingface.py output ===
# Data tests catch silent corruption - wrong label mapping, empty texts slipping through, accidental duplicates
# These are bugs that won't throw errors during training but silently degrade the model

class TestHuggingFaceData:
    def setup_method(self):
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.df = pd.read_csv(os.path.join(data_dir, "raw_hf.csv"))

    def test_required_columns_exist(self):
        """Dataset must have text, label and source columns"""
        assert "text" in self.df.columns
        assert "label" in self.df.columns
        assert "source" in self.df.columns

    def test_no_empty_texts(self):
        """No text field should be empty or NaN"""
        assert self.df["text"].isna().sum() == 0
        assert (self.df["text"].str.strip() == "").sum() == 0

    def test_valid_label_values(self):
        """Labels must only be positive, negative or neutral"""
        valid_labels = {"positive", "negative", "neutral"}
        actual_labels = set(self.df["label"].unique())
        assert actual_labels.issubset(valid_labels), \
            f"Unexpected labels found: {actual_labels - valid_labels}"

    def test_minimum_text_length(self):
        """All texts should be longer than 20 characters"""
        short_texts = self.df[self.df["text"].str.len() <= 20]
        assert len(short_texts) == 0, \
            f"Found {len(short_texts)} texts under 20 characters"

    def test_no_duplicate_texts(self):
        """No duplicate text entries"""
        duplicates = self.df["text"].duplicated().sum()
        assert duplicates == 0, f"Found {duplicates} duplicate texts"

    def test_all_three_classes_present(self):
        """Dataset must contain all three sentiment classes"""
        assert "positive" in self.df["label"].values
        assert "negative" in self.df["label"].values
        assert "neutral" in self.df["label"].values

    def test_class_balance_not_extreme(self):
        """No single class should exceed 60% of the dataset"""
        class_ratios = self.df["label"].value_counts(normalize=True)
        for label, ratio in class_ratios.items():
            assert ratio < 0.60, \
                f"Class '{label}' is {ratio:.1%} of data — too imbalanced"


# ── Tests for build_dataset.py output ─────────────────────────────────────────
class TestBuiltDataset:
    """
    Tests for the built HuggingFace DatasetDict.
    Requires running the full data pipeline locally first:
        python data/build_dataset.py
    Not run in CI : dataset files are not committed to the repo.
    """
    def setup_method(self):
        from datasets import load_from_disk
        data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
        self.dataset = load_from_disk(os.path.join(data_dir, "sg_sentiment_dataset"))

    def test_all_splits_present(self):
        """Dataset must have train, validation and test splits"""
        assert "train" in self.dataset
        assert "validation" in self.dataset
        assert "test" in self.dataset

    def test_split_sizes_are_reasonable(self):
        """Train should be largest split, val and test roughly equal"""
        train_size = len(self.dataset["train"])
        val_size = len(self.dataset["validation"])
        test_size = len(self.dataset["test"])
        assert train_size > val_size
        assert train_size > test_size
        assert abs(val_size - test_size) < 10, \
            "Validation and test splits should be roughly equal"

    def test_required_features_exist(self):
        """Each split must have text, label and source features"""
        for split in ["train", "validation", "test"]:
            features = self.dataset[split].features
            assert "text" in features, f"Missing 'text' in {split}"
            assert "label" in features, f"Missing 'label' in {split}"

    def test_labels_are_integers(self):
        """Labels must be integers 0, 1, 2 after mapping"""
        for split in ["train", "validation", "test"]:
            labels = self.dataset[split]["label"]
            for label in labels:
                assert label in [0, 1, 2], \
                    f"Invalid label {label} found in {split} split"

    def test_no_empty_texts_in_splits(self):
        """No empty texts in any split"""
        for split in ["train", "validation", "test"]:
            texts = self.dataset[split]["text"]
            for text in texts:
                assert text is not None and len(text.strip()) > 0, \
                    f"Empty text found in {split} split"

    def test_total_size_matches_sources(self):
        """Total dataset size should be sum of all splits"""
        total = sum(len(self.dataset[s]) for s in ["train", "validation", "test"])
        assert total > 20000, f"Expected 20k+ examples, got {total}"