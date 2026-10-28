# syntax=docker/dockerfile:1
# Production RAG API (FastAPI). Build:  docker build -t production-rag-api .
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PIP_DEFAULT_TIMEOUT=180 \
    PIP_RETRIES=8 \
    HF_HOME=/home/app/.cache/huggingface

# tesseract: OCR for scanned PDFs · libgomp1: torch runtime
# procps: `ps`, which the parser sandbox uses to watch the worker's memory
RUN apt-get update \
 && apt-get install -y --no-install-recommends tesseract-ocr libgomp1 procps \
 && rm -rf /var/lib/apt/lists/*

RUN useradd --create-home --uid 10001 app \
 && mkdir -p /var/log/rag /var/lib/rag/raw /home/app/.cache/huggingface \
 && chown -R app:app /var/log/rag /var/lib/rag /home/app

WORKDIR /app

# CPU-only torch first: the default x86 wheel drags in multi-GB CUDA libraries.
# Its own layer, so changing requirements.txt does not download it again.
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY --chown=app:app . .
USER app

# Bake the embedding and reranker models in so the container starts offline.
# Best effort: if the download fails here the API fetches them on first use.
RUN python -c "from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('all-MiniLM-L6-v2'); \
CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')" \
 || echo "WARNING: model pre-download failed; models will download on first use."

EXPOSE 8000
HEALTHCHECK --interval=15s --timeout=5s --start-period=60s --retries=5 \
  CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=3).status == 200 else 1)"

CMD ["uvicorn", "online_pipeline.api.gateway:app", "--host", "0.0.0.0", "--port", "8000"]
