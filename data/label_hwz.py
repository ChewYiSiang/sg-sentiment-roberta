import ollama
import pandas as pd
import os

PROMPT_TEMPLATE = """Classify the sentiment of this text as exactly one of: positive, negative, neutral.
Reply with only the single word label. No explanation, no punctuation.

Text: {text}"""

def classify_sentiment(text: str) -> str:
    try:
        response = ollama.generate(
            model="gemma2:2b",
            prompt=PROMPT_TEMPLATE.format(text=text[:512]),
            options={"temperature": 0, "num_predict": 5}
        )
        label = response["response"].strip().lower()

        # strip punctuation in case model adds a period
        label = label.replace(".", "").replace(",", "").strip()

        # debug: uncomment this line if you want to see raw output
        # print(f"  RAW: {repr(response['response'])}")

        return label if label in ["positive", "negative", "neutral"] else "neutral"

    except Exception as e:
        print(f"  Error: {e}")
        return "neutral"

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv("data/raw_hwz.csv")
print(f"Total rows to label: {len(df)}")

# ── Checkpoint logic ───────────────────────────────────────────────────────────
checkpoint_path = "data/hwz_checkpoint.csv"
start_idx = 0

if os.path.exists(checkpoint_path):
    checkpoint_df = pd.read_csv(checkpoint_path)
    start_idx = len(checkpoint_df)
    labels = checkpoint_df["label"].tolist()
    print(f"Checkpoint found — resuming from index {start_idx}")
else:
    labels = []
    print("No checkpoint found — starting from index 0")

# ── Label loop ─────────────────────────────────────────────────────────────────
for i, text in enumerate(df["text"]):
    if i < start_idx:
        continue

    if i % 20 == 0:
        remaining = len(df) - i
        print(f"Progress: {i}/{len(df)} | Remaining: {remaining} posts")

    labels.append(classify_sentiment(str(text)))

    # Save checkpoint every 50 labels
    if i % 50 == 0 and i > 0:
        checkpoint = df.iloc[:len(labels)].copy()
        checkpoint["label"] = labels
        checkpoint.to_csv(checkpoint_path, index=False)

# ── Save final output ──────────────────────────────────────────────────────────
df["label"] = labels
df.to_csv("data/labelled_hwz.csv", index=False)

print("\nHWZ class distribution:")
print(df["label"].value_counts())
print("\nHWZ class percentages:")
print(df["label"].value_counts(normalize=True).round(3) * 100)