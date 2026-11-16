# Serving image for Hugging Face Spaces (Docker SDK) or any container host.
# Models are baked in at build time; the corpus, HNSW index and BM25 matrix (about 2.6 GB)
# are downloaded from the dataset repo on first start and verified against the manifest.
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface \
    HF_HUB_DISABLE_TELEMETRY=1 \
    HF_HUB_DISABLE_SYMLINKS_WARNING=1 \
    TOKENIZERS_PARALLELISM=false \
    OMP_NUM_THREADS=2

RUN useradd -m -u 1000 user
WORKDIR /app

COPY pyproject.toml README.md ./
COPY src ./src
COPY static ./static
COPY evals ./evals

RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install . \
 && python -c "from sentence_transformers import CrossEncoder, SentenceTransformer; CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2', device='cpu'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')" \
 && mkdir -p /app/data/artifacts \
 && chown -R user:user /app

USER user
ENV ARTIFACTS_DIR=/app/data/artifacts \
    PORT=7860
EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=5s --start-period=900s --retries=3 \
  CMD python -c "import urllib.request,sys; r=urllib.request.urlopen('http://127.0.0.1:7860/health'); sys.exit(0 if r.status==200 else 1)"

CMD ["uvicorn", "ragdemo.api:app", "--host", "0.0.0.0", "--port", "7860"]
