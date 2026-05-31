import os
import json
import numpy as np
import torch
import wandb
from datasets import load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from peft import (
    get_peft_model,
    LoraConfig,
    TaskType,
    prepare_model_for_kbit_training,
)
from transformers import BitsAndBytesConfig
from sklearn.metrics import accuracy_score, f1_score, classification_report

# === Config ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "..", "data")

MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment-latest"
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
DATASET_PATH = os.path.join(DATA_DIR, "sg_sentiment_dataset")
WEIGHTS_PATH = os.path.join(DATA_DIR, "class_weights.json")

LABEL2ID = {"negative": 0, "neutral": 1, "positive": 2}
ID2LABEL = {0: "negative", 1: "neutral", 2: "positive"}

# === Load class weights ===
with open(WEIGHTS_PATH) as f:
    class_weights_dict = json.load(f)

class_weights = torch.tensor(
    [class_weights_dict["0"], class_weights_dict["1"], class_weights_dict["2"]],
    dtype=torch.float32
)
print(f"Class weights: {class_weights}")

# === QLoRA config ===
# 4-bit quantisation — reduces VRAM usage by ~4x, enabling fine-tuning on 8GB GPU
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,        # nested quantisation saves extra memory
    bnb_4bit_quant_type="nf4",             # NormalFloat4 - best for normally distributed weights
    bnb_4bit_compute_dtype=torch.bfloat16  # compute in bfloat16 for stability
)

# === LoRA config ===
# Adds small adapter matrices to attention layers - trains ~1% of total parameters
# r=16: rank controls adapter capacity - sweet spot for 3-class classification
# lora_alpha=32: scaling factor - standard practice to set at 2x rank
lora_config = LoraConfig(
    task_type=TaskType.SEQ_CLS,
    r=8,
    lora_alpha=16,
    lora_dropout=0.1,
    bias="none",
    target_modules=["query", "value"],
)

# === Load tokenizer ===
print(f"Loading tokenizer: {MODEL_NAME}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

# === Load model ===
print(f"Loading model in 4-bit: {MODEL_NAME}")
model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=3,
    id2label=ID2LABEL,
    label2id=LABEL2ID,
    quantization_config=bnb_config,
    device_map="auto",
)

# === Fix: cast classifier head to float32 ===
# 4-bit tensors cannot hold gradients — classifier head must stay in float32
for param in model.classifier.parameters():
    param.data = param.data.to(torch.float32)

# Only cast pooler if it exists and is not None
if hasattr(model, "roberta") and model.roberta.pooler is not None:
    for param in model.roberta.pooler.parameters():
        param.data = param.data.to(torch.float32)

# === Prepare for k-bit training ===
model = prepare_model_for_kbit_training(model)

# === Apply LoRA adapters ===
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# === Load and tokenize dataset ===
print("Loading dataset...")
dataset = load_from_disk(DATASET_PATH)

def tokenize(batch):
    return tokenizer(
        batch["text"],
        truncation=True,
        max_length=128,  # tweets and forum posts are short — 128 covers 99% of data
        padding=False,   # DataCollatorWithPadding handles this dynamically per batch
    )

print("Tokenizing dataset...")
tokenized = dataset.map(tokenize, batched=True, remove_columns=["text", "source"])
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# === Custom Trainer with class weights ===
# Standard Trainer uses equal loss — this penalises minority class errors more
class WeightedTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits

        # Cast logits to float32 — needed because model computes in bfloat16
        # but CrossEntropyLoss expects float32
        loss_fn = torch.nn.CrossEntropyLoss(
            weight=class_weights.to(logits.device)
        )
        loss = loss_fn(logits.float(), labels)
        return (loss, outputs) if return_outputs else loss

# === Metrics ===
def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    acc = accuracy_score(labels, predictions)
    f1_macro = f1_score(labels, predictions, average="macro")
    f1_weighted = f1_score(labels, predictions, average="weighted")
    return {
        "accuracy": acc,
        "f1_macro": f1_macro,
        "f1_weighted": f1_weighted,
    }

# === Training arguments ===
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=3,
    per_device_train_batch_size=32,
    per_device_eval_batch_size=64,
    learning_rate=5e-4,
    warmup_steps=200,             # warm up LR for first 200 steps - avoids large updates early
    weight_decay=0.01,            # L2 regularisation - prevents overfitting
    eval_strategy="epoch",        # evaluate at end of each epoch
    save_strategy="epoch",        # save checkpoint each epoch
    load_best_model_at_end=True,  # keep best checkpoint based on f1_macro
    metric_for_best_model="f1_macro",
    greater_is_better=True,
    logging_steps=50,
    fp16=True,                    # mixed precision - faster on RTX 4060 Tensor Cores
    report_to="wandb",
    run_name="sg-sentiment-qlora-r8-lr5e4",
)

# === Initialise W&B run ===
wandb.init(
    project="sg-sentiment-roberta",
    name="qlora-r8-lr5e4-run3",
    config={
        "model": MODEL_NAME,
        "lora_r": 8,
        "lora_alpha": 16,
        "learning_rate": 5e-4,
        "epochs": 3,
        "batch_size": 32,
        "quantisation": "4bit-nf4",
    }
)

# === Train ===
trainer = WeightedTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized["train"],
    eval_dataset=tokenized["validation"],
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print("Starting training...")
trainer.train()

# === Evaluate on test set ===
print("\nEvaluating on test set...")
predictions = trainer.predict(tokenized["test"])
preds = np.argmax(predictions.predictions, axis=-1)
labels = predictions.label_ids

print("\n=== Test Set Results ===")
print(f"Accuracy:    {accuracy_score(labels, preds):.4f}")
print(f"F1 Macro:    {f1_score(labels, preds, average='macro'):.4f}")
print(f"F1 Weighted: {f1_score(labels, preds, average='weighted'):.4f}")
print("\nPer-class breakdown:")
print(classification_report(labels, preds, target_names=["negative", "neutral", "positive"]))

# === Log final test metrics to W&B ===
wandb.log({
    "test/accuracy": accuracy_score(labels, preds),
    "test/f1_macro": f1_score(labels, preds, average="macro"),
    "test/f1_weighted": f1_score(labels, preds, average="weighted"),
})

wandb.finish()
print(f"\nTraining complete. Model saved to {OUTPUT_DIR}")