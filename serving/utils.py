import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "..", "training", "output", "final_adapter_v2")

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}


def load_model_and_tokenizer():
    """
    Load the QLoRA fine-tuned model and tokenizer.
    Centralised here so FastAPI, Gradio and tests all use identical loading logic.
    """
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

    # Classifier head must be float32 — 4-bit tensors cannot hold gradients

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    return model, tokenizer


def predict(text: str, model, tokenizer) -> dict:
    """
    Run inference on a single text.
    Returns label, confidence and per-class probabilities.
    """
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    inputs = {k: v.cuda() for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)

    probs = torch.nn.functional.softmax(outputs.logits, dim=-1)
    predicted_id = probs.argmax().item()
    confidence = probs.max().item()

    return {
        "label": ID2LABEL[predicted_id],
        "confidence": round(confidence, 4),
        "probabilities": {
            ID2LABEL[i]: round(probs[0][i].item(), 4)
            for i in range(3)
        }
    }