# ---- Base Image (lightweight + stable for FAISS & transformers)
FROM python:3.10-slim

# Environment settings (important for production logs)
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Set working directory inside container
WORKDIR /app

# Install system dependencies (CRITICAL for your project)
# git -> needed for GitHub repo ingestion
# build-essential -> needed for faiss & some python wheels
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first (for better Docker caching)
COPY requirements.txt .

# Upgrade pip and install Python dependencies
RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir -r requirements.txt

# Copy entire project (app, ui, main.py)
COPY . .

# Create persistent directories for RAG storage
# (prevents runtime folder errors)
RUN mkdir -p repos indexes

# Expose FastAPI port
EXPOSE 8000

# Start FastAPI (production safe)
# 0.0.0.0 is REQUIRED for Docker hosting
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]