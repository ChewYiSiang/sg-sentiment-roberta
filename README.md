# SG Sentiment RoBERTa

Fine-tuned sentiment classifier on Singapore-specific text using QLoRA,
with experiment tracking via Weights & Biases and deployment via FastAPI + Docker.

## Project motivation
Most pre-trained sentiment models are trained on American English Twitter data.
This project fine-tunes a transformer on locally scraped Singaporean forum text
(Hardwarezone) augmented with a public tweet sentiment dataset, improving
generalisation on Singlish and local language patterns.

## Results

| Model | Accuracy | F1 Macro | Params trained | Notes |
|---|---|---|---|---|
| TF-IDF + Logistic Regression (baseline) | 0.89* | - | - | Lazada reviews, binary classification |
| QLoRA RoBERTa r=16, lr=2e-4 | 0.768 | 0.770 | 2.96M | Run 1 |
| QLoRA RoBERTa r=8, lr=2e-4 | 0.766 | 0.768 | 1.48M | Run 2 |
| QLoRA RoBERTa r=8, lr=5e-4 | **0.774** | **0.776** | **1.48M** | Run 3 — best |

*Baseline is not directly comparable — different dataset and binary vs 3-class task

## Architecture
(diagram to be added in Phase 5)

## Stack
- Fine-tuning: HuggingFace Transformers, PEFT, QLoRA, bitsandbytes
- Tracking: Weights & Biases
- Serving: FastAPI, Docker, GitHub Actions
- Data: HuggingFace datasets, Hardwarezone (scraped), Gemma2:2b labeller

## Quickstart
\`\`\`bash
git clone https://github.com/ChewYiSiang/sg-sentiment-roberta
cd sg-sentiment-roberta
python -m venv venv && source venv/bin/activate
pip install -r requirements.txt
\`\`\`

## Project structure
\`\`\`
sg-sentiment-roberta/
├── data/          # data collection and labelling scripts
├── training/      # fine-tuning scripts
├── serving/       # FastAPI inference server
├── tests/         # unit tests
└── .github/       # CI/CD workflows
\`\`\`

## Reproducing the data pipeline
\`\`\`bash
python data/scrape_huggingface.py
python data/scrape_hwz.py
python data/label.py
python data/build_dataset.py
\`\`\`