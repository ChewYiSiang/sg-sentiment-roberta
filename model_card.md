# SG Sentiment RoBERTa

Sentiment classifier fine-tuned on Singapore-specific text using QLoRA.

## Model details
- Base model: cardiffnlp/twitter-roberta-base-sentiment-latest
- Fine-tuning method: QLoRA (4-bit NF4 + LoRA r=8, alpha=16)
- Task: 3-class sentiment classification (positive / negative / neutral)
- Best config: r=8, lr=5e-4

## Training data
- HuggingFace tweet sentiment dataset (25,342 examples)
- Hardwarezone forum posts scraped and labelled with Gemma2:2b (1,998 examples)
- Total: 27,340 examples across train/val/test splits

## Results
| Metric | Score |
|---|---|
| Accuracy | 0.7736 |
| F1 Macro | 0.7759 |
| F1 Weighted | 0.7707 |

## Experiment comparison
| Run | Config | F1 Macro |
|---|---|---|
| Run 1 | r=16, lr=2e-4 | 0.770 |
| Run 2 | r=8, lr=2e-4 | 0.768 |
| Run 3 | r=8, lr=5e-4 | **0.776** |

## Intended use
Sentiment analysis of Singapore English and Singlish text.

## Limitations
- Labels generated via Gemma2:2b for HWZ data — may contain noise
- Limited Singlish coverage in training data
- Not suitable for formal document sentiment analysis