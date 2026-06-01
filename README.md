# SG Sentiment RoBERTa

Fine-tuned sentiment classifier on Singapore-specific text using QLoRA,
with experiment tracking via Weights & Biases and deployment via FastAPI + Docker.

![Tests](https://github.com/ChewYiSiang/sg-sentiment-roberta/actions/workflows/test.yml/badge.svg)

## Live demo
[Try it on HuggingFace Spaces](https://huggingface.co/spaces/Yi-Siang/sg-sentiment-roberta)

## Project motivation
Most pre-trained sentiment models are trained on American English Twitter data.
This project fine-tunes a transformer on locally scraped Singaporean forum text
(Hardwarezone) augmented with a public tweet sentiment dataset, improving
generalisation on Singlish and local language patterns.

## Results

| Model | Accuracy | F1 Macro | Params trained | Notes |
|---|---|---|---|---|
| QLoRA RoBERTa r=16, lr=2e-4 | 0.768 | 0.770 | 2.96M | Run 1 |
| QLoRA RoBERTa r=8, lr=2e-4 | 0.766 | 0.768 | 1.48M | Run 2 |
| QLoRA RoBERTa r=8, lr=5e-4 | 0.774 | 0.776 | 1.48M | Run 3 |
| QLoRA RoBERTa r=8, lr=5e-4 (fixed classifier) | **0.782** | **0.784** | **1.48M** | Run 4 — best |

## Architecture

```
Raw data (HuggingFace tweets + Hardwarezone forum posts)
        ↓
Labelling (Gemma2:2b local LLM)
        ↓
HuggingFace DatasetDict (train / val / test, stratified)
        ↓
QLoRA fine-tuning (RoBERTa + LoRA adapters, 4-bit NF4)
        ↓
Experiment tracking (Weights & Biases — 4 runs)
        ↓
Merged float32 model → HuggingFace Hub
        ↓
FastAPI serving (/predict endpoint) → Docker container
        ↓
Gradio demo → HuggingFace Spaces (live public URL)
```

## Stack
- **Fine-tuning:** HuggingFace Transformers, PEFT, QLoRA, bitsandbytes
- **Experiment tracking:** Weights & Biases
- **Serving:** FastAPI, uvicorn, Docker, docker-compose
- **CI/CD:** GitHub Actions (runs 20 tests on every push)
- **Data:** HuggingFace datasets, Hardwarezone (scraped), Gemma2:2b labeller
- **Testing:** pytest (30 tests across data pipeline, preprocessing, inference)
- **Demo:** Gradio on HuggingFace Spaces

## Quickstart
```bash
git clone https://github.com/ChewYiSiang/sg-sentiment-roberta
cd sg-sentiment-roberta
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Reproducing the data pipeline
```bash
python data/scrape_huggingface.py   # pull public tweet sentiment dataset
python data/scrape_hwz.py           # scrape Hardwarezone forum posts
python data/label.py                # label HWZ posts with Gemma2:2b
python data/build_dataset.py        # merge, split and save dataset
```

## Running the API locally
```bash
python serving/app.py
# API available at http://localhost:8000
# Interactive docs at http://localhost:8000/docs
```

## Running tests
```bash
pytest tests/ -v
```

## Project structure
```
sg-sentiment-roberta/
├── .github/
│   └── workflows/
│       ├── test.yml              # runs pytest on every push
│       └── docker-publish.yml    # builds and pushes Docker image on release tag
├── data/
│   ├── scrape_huggingface.py     # pull public sentiment dataset
│   ├── scrape_hwz.py             # scrape Hardwarezone forum posts
│   ├── label.py                  # label with Gemma2:2b local LLM
│   └── build_dataset.py          # merge, stratified split, save
├── training/
│   ├── train.py                  # QLoRA fine-tuning with W&B tracking
│   ├── merge_and_save.py         # merge LoRA into base model for deployment
│   └── push_to_hub.py            # publish to HuggingFace Hub
├── serving/
│   ├── utils.py                  # shared model loading logic
│   ├── app.py                    # FastAPI REST endpoint
│   ├── demo.py                   # Gradio demo (local)
│   ├── app_spaces.py             # Gradio demo (HuggingFace Spaces)
│   └── Dockerfile                # containerised inference server
├── tests/
│   ├── test_data.py              # 11 data pipeline tests
│   ├── test_preprocess.py        # 9 tokenizer and preprocessing tests
│   └── test_inference.py         # 10 end-to-end inference tests
├── docker-compose.yml
├── requirements.txt
├── model_card.md
└── README.md
```

## Model
Published on HuggingFace Hub: [Yi-Siang/sg-sentiment-roberta](https://huggingface.co/Yi-Siang/sg-sentiment-roberta)
```python
from transformers import pipeline
classifier = pipeline("sentiment-analysis", model="Yi-Siang/sg-sentiment-roberta")
classifier("wah shiok sia this chicken rice confirm best")
```