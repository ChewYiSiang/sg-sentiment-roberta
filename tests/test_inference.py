import pytest
import os
import sys
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ADAPTER_PATH = os.path.join(BASE_DIR, "..", "training", "output", "final_adapter_v2")
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# Inference tests verify the deployed model behaves correctly end to end,
# valid labels, probabilities that sum to 1, no crashes on weird input.
# The test_obvious_positive and test_obvious_negative tests are particularly important

@pytest.fixture(scope="module")
def model_and_tokenizer():
    """Load model and tokenizer once for all inference tests"""
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )

    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
        quantization_config=bnb_config,
        device_map="auto",
    )


    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    # Debug: print device of every parameter
    for name, param in model.named_parameters():
        print(f"{name}: {param.device} | {param.dtype}")

    return model, tokenizer

def predict(text: str, model, tokenizer) -> dict:
    """Helper: run inference on a single text"""
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    # Move inputs to GPU directly — model is on cuda:0 via device_map="auto"
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = probs.argmax().item()
    confidence = probs.max().item()

    return {
        "label": ID2LABEL[predicted_id],
        "confidence": confidence,
        "probabilities": {
            ID2LABEL[i]: probs[0][i].item()
            for i in range(3)
        }
    }


class TestInference:
    def test_model_loads_without_error(self, model_and_tokenizer):
        """Model and tokenizer should load successfully"""
        model, tokenizer = model_and_tokenizer
        assert model is not None
        assert tokenizer is not None

    def test_output_shape(self, model_and_tokenizer):
        """Model should output logits with 3 classes"""
        model, tokenizer = model_and_tokenizer
        inputs = tokenizer("test text", return_tensors="pt")
        inputs = {k: v.cuda() for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)
        assert outputs.logits.shape[-1] == 3

    def test_returns_valid_label(self, model_and_tokenizer):
        """Prediction must be one of three valid labels"""
        model, tokenizer = model_and_tokenizer
        result = predict("This is amazing!", model, tokenizer)
        assert result["label"] in ["positive", "negative", "neutral"]

    def test_confidence_between_0_and_1(self, model_and_tokenizer):
        """Confidence score must be a valid probability"""
        model, tokenizer = model_and_tokenizer
        result = predict("Good product", model, tokenizer)
        assert 0.0 <= result["confidence"] <= 1.0

    def test_probabilities_sum_to_1(self, model_and_tokenizer):
        """All class probabilities must sum to 1"""
        model, tokenizer = model_and_tokenizer
        result = predict("Okay experience", model, tokenizer)
        total = sum(result["probabilities"].values())
        assert abs(total - 1.0) < 1e-5, \
            f"Probabilities sum to {total}, expected 1.0"

    def test_obvious_positive(self, model_and_tokenizer):
        """Clearly positive text should be classified as positive"""
        model, tokenizer = model_and_tokenizer
        result = predict("This is absolutely fantastic, best ever!", model, tokenizer)
        assert result["label"] == "positive", \
            f"Expected positive, got {result['label']}"

    def test_obvious_negative(self, model_and_tokenizer):
        """Clearly negative text should be classified as negative"""
        model, tokenizer = model_and_tokenizer
        result = predict("Terrible, worst experience, total waste of money", model, tokenizer)
        assert result["label"] == "negative", \
            f"Expected negative, got {result['label']}"

    def test_empty_string_does_not_crash(self, model_and_tokenizer):
        """Model should not crash on empty input"""
        model, tokenizer = model_and_tokenizer
        try:
            result = predict("", model, tokenizer)
            assert result["label"] in ["positive", "negative", "neutral"]
        except Exception as e:
            pytest.fail(f"Model crashed on empty string: {e}")

    def test_long_text_truncated_correctly(self, model_and_tokenizer):
        """Very long text should not crash — truncation handles it"""
        model, tokenizer = model_and_tokenizer
        long_text = "good " * 500
        try:
            result = predict(long_text, model, tokenizer)
            assert result["label"] in ["positive", "negative", "neutral"]
        except Exception as e:
            pytest.fail(f"Model crashed on long text: {e}")

    def test_singlish_input(self, model_and_tokenizer):
        """Singlish text should return a valid prediction"""
        model, tokenizer = model_and_tokenizer
        result = predict("wah shiok sia this chicken rice confirm best in sg", model, tokenizer)
        assert result["label"] in ["positive", "negative", "neutral"]
        assert result["confidence"] > 0.0