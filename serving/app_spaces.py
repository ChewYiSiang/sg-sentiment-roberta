import gradio as gr
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
ADAPTER_REPO = "ChewYiSiang/sg-sentiment-roberta"  # your HuggingFace repo

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL_EMOJI = {
    "positive": "😊 Positive",
    "negative": "😞 Negative",
    "neutral":  "😐 Neutral"
}

print("Loading model...")

# On Spaces we may not have a GPU — handle both cases
device = "cuda" if torch.cuda.is_available() else "cpu"

if device == "cuda":
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
    for param in base_model.classifier.parameters():
        param.data = param.data.to(dtype=torch.float32, device="cuda")
else:
    # CPU fallback — no quantisation, slower but works anywhere
    base_model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME,
        num_labels=3,
    )

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = PeftModel.from_pretrained(base_model, ADAPTER_REPO)
model.eval()
print(f"Model loaded on {device}")


def predict(text: str) -> dict:
    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=128,
    )
    if device == "cuda":
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


def classify(text: str):
    if not text or not text.strip():
        return "Please enter some text", {}, ""
    result = predict(text)
    label = LABEL_EMOJI[result["label"]]
    confidence = f"{result['confidence']*100:.1f}%"
    probs = {LABEL_EMOJI[k]: v for k, v in result["probabilities"].items()}
    return label, probs, confidence


demo = gr.Interface(
    fn=classify,
    inputs=gr.Textbox(
        lines=3,
        placeholder="Enter text to classify... (English or Singlish)",
        label="Input text"
    ),
    outputs=[
        gr.Textbox(label="Sentiment"),
        gr.Label(label="Confidence breakdown"),
        gr.Textbox(label="Confidence score"),
    ],
    title="SG Sentiment Classifier",
    description=(
        "Fine-tuned RoBERTa on Singapore text using QLoRA. "
        "Trained on HuggingFace tweet sentiment + Hardwarezone forum posts."
    ),
    examples=[
        ["wah shiok sia this chicken rice confirm best in singapore"],
        ["terrible service, total waste of money, never coming back"],
        ["the shop opens at 9am on weekdays"],
        ["so sian lah this queue very long"],
        ["best purchase ever, highly recommend to everyone!"],
    ],
)

if __name__ == "__main__":
    demo.launch(theme=gr.themes.Soft())