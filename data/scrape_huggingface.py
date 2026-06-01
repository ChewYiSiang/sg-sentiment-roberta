from datasets import load_dataset
import pandas as pd

# this dataset is one of the most widely used benchmark sentiment datasets. 
# it's clean, well-labelled, and Twitter text is closer to forum posts than formal text
print("Loading dataset from HuggingFace...")
dataset = load_dataset("mteb/tweet_sentiment_extraction", split="train")
df = pd.DataFrame(dataset)

print("Columns available:", df.columns.tolist())
print("Label values found:", df["label"].unique())
print("\nRaw label distribution:")
print(df["label"].value_counts())

# mteb/tweet_sentiment_extraction uses integer labels: 0=negative, 1=neutral, 2=positive
label_map = {0: "negative", 1: "neutral", 2: "positive"}
df["label"] = df["label"].map(label_map)

# keep only what we need
df = df[["text", "label"]].copy()

# filter very short texts, too little signal
df = df[df["text"].str.len() > 20]
df = df.dropna()
df["source"] = "huggingface_tweet"

df.to_csv("data/raw_hf.csv", index=False)

print(f"\nFinal HuggingFace examples: {len(df)}")
print("\nMapped label distribution:")
print(df["label"].value_counts())
print("\nLabel percentages:")
print(df["label"].value_counts(normalize=True).round(3) * 100)