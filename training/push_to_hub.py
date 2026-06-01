import os
import sys
from huggingface_hub import HfApi
from transformers import AutoTokenizer
from peft import PeftModel, AutoPeftModelForSequenceClassification

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ADAPTER_PATH = os.path.join(BASE_DIR, "output", "final_adapter")
HF_REPO = "Yi-Siang/sg-sentiment-roberta"

print(f"Pushing adapter to {HF_REPO}...")

# Push adapter weights and config
api = HfApi()
api.create_repo(HF_REPO, exist_ok=True)
api.upload_folder(
    folder_path=ADAPTER_PATH,
    repo_id=HF_REPO,
    repo_type="model"
)

# Push tokenizer
tokenizer = AutoTokenizer.from_pretrained(
    "cardiffnlp/twitter-roberta-base-sentiment-latest"
)
tokenizer.push_to_hub(HF_REPO)

print(f"Done — model at https://huggingface.co/{HF_REPO}")