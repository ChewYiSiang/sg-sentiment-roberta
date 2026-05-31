import gradio as gr
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.utils import load_model_and_tokenizer, predict

# Load model once at startup
print("Loading model...")
model, tokenizer = load_model_and_tokenizer()
print("Model ready")

LABEL_EMOJI = {
    "positive": "😊 Positive",
    "negative": "😞 Negative",
    "neutral":  "😐 Neutral"
}

def classify(text: str):
    if not text or not text.strip():
        return "Please enter some text", {}, ""

    result = predict(text, model, tokenizer)
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