# ---- Lightweight base (best balance for transformers + FAISS)
FROM python:3.10-slim

# ---- Environment (IMPORTANT for streaming + logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV PIP_NO_CACHE_DIR=1
ENV PYTHONIOENCODING=UTF-8

# ---- Working directory
WORKDIR /app

# ---- System deps (required for your project)
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# ---- Copy requirements first (Docker cache optimization)
COPY requirements.txt .

# ---- Install dependencies (NO CACHE = saves GBs of space)
RUN pip install --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ---- Copy full project
COPY . .

# ---- Create persistent dirs (RAG indexing safety)
RUN mkdir -p /app/repos /app/indexes

# ---- Expose FastAPI port
EXPOSE 8000

# ---- Production start (CRITICAL for streaming)
# h11 = disables buffering (fixes streaming issue you faced)
CMD ["uvicorn", "main:app", \
     "--host", "0.0.0.0", \
     "--port", "8000", \
     "--http", "h11", \
     "--workers", "1"]