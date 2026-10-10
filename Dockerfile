# ==============================================================
# Multimodal Clinical Decision Support System — Dockerfile
# Multi-stage build: backend (FastAPI) & frontend (Streamlit)
# ==============================================================

# ---- Base Stage ----
FROM python:3.13-slim AS base

# Prevent Python from writing .pyc files and enable unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Install system dependencies required by asyncpg, opencv, and torch
RUN apt-get update && apt-get install -y --no-install-recommends \
    libpq-dev \
    libgl1 \
    libglib2.0-0 \
    curl \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# ---- Dependencies Stage ----
FROM base AS deps

# Copy only requirements first for better layer caching
COPY requirements.txt .

# Install Python dependencies with CPU-only PyTorch
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements.txt

# ---- Backend Target ----
FROM deps AS backend

# Copy application source code
COPY app/ ./app/
COPY data/guidelines/ ./data/guidelines/
COPY run.py .

# Create necessary directories
RUN mkdir -p storage/images storage/reports storage/logs data/chroma_db data/sample_images models

EXPOSE 8008

# Long start period: the first start downloads the embedding model and
# builds the guideline index before the server accepts requests.
HEALTHCHECK --interval=30s --timeout=10s --start-period=300s --retries=3 \
    CMD curl -f http://localhost:8008/ || exit 1

CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8008"]

# ---- Frontend Target ----
# Built on the base image, not on the backend's dependency stage: the UI
# talks to the API over HTTP and imports nothing from app/, so it needs
# Streamlit, an HTTP client and Pillow, not torch or the vector store.
FROM base AS frontend

COPY requirements-frontend.txt .
RUN pip install --no-cache-dir -r requirements-frontend.txt

# Copy Streamlit application
COPY streamlit_app/ ./streamlit_app/

# Create Streamlit config to disable CORS/XSRF for Docker networking
RUN mkdir -p /root/.streamlit && \
    printf '[server]\nheadless = true\nport = 8501\nenableCORS = false\nenableXsrfProtection = false\n\n[theme]\nbase = "dark"\n' \
    > /root/.streamlit/config.toml

EXPOSE 8501

HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD curl -f http://localhost:8501/_stcore/health || exit 1

CMD ["python", "-m", "streamlit", "run", "streamlit_app/app.py"]
