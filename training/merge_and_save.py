import os
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, BitsAndBytesConfig
from peft import PeftModel
from huggingface_hub import HfApi

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "output", "final_adapter_v2")
MERGED_PATH = os.path.join(BASE_DIR, "output", "merged_model")
HF_REPO = "Yi-Siang/sg-sentiment-roberta"
MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"

ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}
LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}

# ── Step 1: Load base model in full float32 on CPU ────────────────────────────
# Deliberately NOT using BitsAndBytesConfig here
# We load clean float32 weights so the merge produces standard tensors
print("Loading base model in float32 on CPU...")
base_model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    torch_dtype=torch.float32,
    device_map="cpu",  # force CPU — no quantisation
)

# ── Step 2: Load LoRA adapter on top ──────────────────────────────────────────
print("Loading LoRA adapter...")
peft_model = PeftModel.from_pretrained(
    base_model,
    ADAPTER_PATH,
    torch_dtype=torch.float32,
    device_map="cpu",
)

# ── Step 3: Merge and unload ───────────────────────────────────────────────────
# Fuses lora_B × lora_A into base weights mathematically
# Returns a plain AutoModelForSequenceClassification with no PEFT dependency
print("Merging adapter into base model weights...")
merged_model = peft_model.merge_and_unload()
print(f"Merged model type: {type(merged_model)}")

# ── Step 4: Save locally ───────────────────────────────────────────────────────
print(f"Saving merged model to {MERGED_PATH}...")
os.makedirs(MERGED_PATH, exist_ok=True)
merged_model.save_pretrained(MERGED_PATH)

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.save_pretrained(MERGED_PATH)
print("Saved successfully")

# ── Step 5: Push to HuggingFace Hub ───────────────────────────────────────────
print(f"\nPushing to {HF_REPO}...")
api = HfApi()
api.upload_folder(
    folder_path=MERGED_PATH,
    repo_id=HF_REPO,
    repo_type="model",
    commit_message="Add merged float32 model for CPU inference"
)
print(f"Done — https://huggingface.co/{HF_REPO}")