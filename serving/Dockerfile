FROM python:3.11-slim

# Install system dependencies for bitsandbytes and torch
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements first: Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY serving/ ./serving/
COPY training/output/final_adapter_v2 ./training/output/final_adapter_v2

# Expose port
EXPOSE 8000

# Health check — Docker will restart container if this fails
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["python", "-m", "uvicorn", "serving.app:app", "--host", "0.0.0.0", "--port", "8000"]