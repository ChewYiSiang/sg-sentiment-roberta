# data/test_label.py
import ollama

PROMPT_TEMPLATE = """Classify the sentiment of this text as exactly one of: positive, negative, neutral.
Reply with only the single word label. No explanation, no punctuation.

Text: {text}"""

test_cases = [
    ("This is amazing, best purchase ever!", "positive"),
    ("Terrible service, total waste of money", "negative"),
    ("The shop opens at 9am on weekdays", "neutral"),
    ("so shiok lah this food confirm best", "positive"),
    ("wah lao this queue so long very sian", "negative"),
]

print("=== Sanity check ===\n")
for text, expected in test_cases:
    response = ollama.generate(
        model="gemma2:2b",
        prompt=PROMPT_TEMPLATE.format(text=text),
        options={"temperature": 0, "num_predict": 5}
    )
    raw = response["response"].strip().lower().replace(".", "").strip()
    status = "✓" if raw == expected else "✗"
    print(f"{status} Expected: {expected:10s} | Got: {raw:10s} | {text[:50]}")