from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from serving.utils import load_model_and_tokenizer, predict

# === Lifespan: load model once at startup ===
# Model loading is expensive (~5s) it will run when the server starts
# and reuse the same model for all requests
model = None
tokenizer = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model, tokenizer
    print("Loading model...")
    model, tokenizer = load_model_and_tokenizer()
    print("Model loaded — server ready")
    yield
    # Cleanup on shutdown
    del model, tokenizer

app = FastAPI(
    title="SG Sentiment API",
    description="Sentiment classifier fine-tuned on Singapore text using QLoRA",
    version="1.0.0",
    lifespan=lifespan
)

# === Request / Response schemas ===
class PredictRequest(BaseModel):
    text: str

    class Config:
        json_schema_extra = {
            "example": {"text": "wah shiok sia this chicken rice confirm best in sg"}
        }

class PredictResponse(BaseModel):
    label: str
    confidence: float
    probabilities: dict

# === Routes ===
@app.get("/health")
def health():
    """Health check — confirms server and model are running"""
    return {"status": "ok", "model_loaded": model is not None}

@app.post("/predict", response_model=PredictResponse)
def predict_sentiment(request: PredictRequest):
    """
    Classify sentiment of input text.
    Returns label (positive/negative/neutral), confidence score,
    and per-class probability breakdown.
    """
    if not request.text or not request.text.strip():
        raise HTTPException(status_code=422, detail="Text cannot be empty")

    if len(request.text) > 5000:
        raise HTTPException(status_code=422, detail="Text too long — max 5000 characters")

    result = predict(request.text, model, tokenizer)
    return result

@app.get("/")
def root():
    return {
        "name": "SG Sentiment API",
        "docs": "/docs",
        "health": "/health",
        "predict": "/predict"
    }

if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=False)