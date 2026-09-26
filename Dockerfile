# Slim runtime image (~512 MB RAM is enough): embeddings on ONNX Runtime, no PyTorch.
FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1 HF_HOME=/app/.hf \
    EMBED_BACKEND=onnx EMBED_THREADS=1 LOCAL_MODEL=lite MAX_UPLOAD_ROWS=20000 MALLOC_ARENA_MAX=2
# Gemini goes through the organisers' proxy. Not secret; GEMINI_API_KEY itself is set as a secret in the host.
# GEMINI_QUOTA_TOTAL = requests left when this image was prepared, so the quota floor works in a fresh container.
ENV GEMINI_BACKEND=hackathon \
    GEMINI_BASE_URL=https://hackathon-api-new-152590733511.northamerica-northeast2.run.app \
    GEMINI_QUOTA_TOTAL=862
WORKDIR /app
COPY requirements-deploy.txt .
RUN pip install -r requirements-deploy.txt
# bake the embedding model (official ONNX export of all-MiniLM-L6-v2) into the image
RUN python -c "from huggingface_hub import hf_hub_download as d; [d('sentence-transformers/all-MiniLM-L6-v2', f) for f in ('onnx/model.onnx', 'tokenizer.json')]"
COPY app app
COPY pipeline pipeline
COPY models models
COPY data/gazetteer data/gazetteer
COPY data/processed/main_results.json.gz data/processed/main_results.meta.json data/processed/main_results.summary.json.gz data/processed/
COPY data/processed/bonus_results.json.gz data/processed/bonus_results.meta.json data/processed/bonus_results.summary.json.gz data/processed/
COPY data/cache/osm data/cache/osm
COPY data/cache/geocode.json data/cache/geocode.json
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
# Render sets $PORT; Hugging Face uses 7860
CMD ["sh", "-c", "uvicorn app.server:app --host 0.0.0.0 --port ${PORT:-7860}"]
