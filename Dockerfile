FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    tini \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# CPU wheel first. Timestral's dependency set would otherwise resolve a CUDA build.
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

COPY config.py engine.py main.py database.py security.py admin.py ./
COPY static ./static

RUN useradd -m -u 1000 appuser && \
    mkdir -p /app/data/huggingface && \
    chown -R appuser:appuser /app

ENV HF_HOME=/app/data/huggingface \
    HUGGINGFACE_HUB_CACHE=/app/data/huggingface \
    TORCH_NUM_THREADS=4

USER appuser

EXPOSE 8003

ENTRYPOINT ["tini", "--"]

# The process accepts traffic only after the checkpoint is loaded, so the
# start period covers the first Hub download (~840 MB for 209M).
HEALTHCHECK --interval=30s --timeout=10s --start-period=600s --retries=5 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8003/health')" || exit 1

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8003", "--log-level", "info"]
