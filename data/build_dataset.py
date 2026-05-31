from datasets import Dataset, DatasetDict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight
import json
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# === Load both sources ===
hf_df = pd.read_csv(os.path.join(BASE_DIR, "raw_hf.csv"))[["text", "label", "source"]]
hwz_df = pd.read_csv(os.path.join(BASE_DIR, "labelled_hwz_clean.csv"))[["text", "label", "source"]]

# === Combine ===
df = pd.concat([hf_df, hwz_df], ignore_index=True)
df = df.dropna(subset=["text", "label"])

print("=== Combined dataset overview ===")
print(f"Total examples: {len(df)}")
print("\nSource breakdown:")
print(df["source"].value_counts())
print("\nOverall label distribution:")
print(df["label"].value_counts())
print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True).round(3) * 100)

# === Map labels to integers ===
label2id = {"negative": 0, "neutral": 1, "positive": 2}
id2label = {0: "negative", 1: "neutral", 2: "positive"}

df["label"] = df["label"].map(label2id)
df = df.dropna(subset=["label"])
df["label"] = df["label"].astype(int)

# === Compute class weights ===
# this tells the trainer to penalise mistakes on minority classes more heavily
classes = np.array([0, 1, 2])
weights = compute_class_weight(
    class_weight="balanced",
    classes=classes,
    y=df["label"].values
)
class_weights = {int(k): float(v) for k, v in zip(classes, weights)}
print(f"\nClass weights: {class_weights}")
print("(weights > 1.0 mean that class is under-represented)")

# save weights so training script can load them
weights_path = os.path.join(BASE_DIR, "class_weights.json")
with open(weights_path, "w") as f:
    json.dump(class_weights, f)
print(f"Class weights saved to {weights_path}")

# === Stratified split ===
train_df, temp_df = train_test_split(
    df, test_size=0.2, stratify=df["label"], random_state=42
)
val_df, test_df = train_test_split(
    temp_df, test_size=0.5, stratify=temp_df["label"], random_state=42
)

print(f"\nSplit sizes:")
print(f"  Train:      {len(train_df)}")
print(f"  Validation: {len(val_df)}")
print(f"  Test:       {len(test_df)}")

# === Build HuggingFace DatasetDict ===
dataset = DatasetDict({
    "train": Dataset.from_pandas(train_df.reset_index(drop=True)),
    "validation": Dataset.from_pandas(val_df.reset_index(drop=True)),
    "test": Dataset.from_pandas(test_df.reset_index(drop=True))
})

output_path = os.path.join(BASE_DIR, "sg_sentiment_dataset")
dataset.save_to_disk(output_path)

print("\n=== Final dataset ===")
print(dataset)
print("\nPhase 1 data pipeline completed - 27340 examples across HF + HWZ sources, ready for Phase 2 training")