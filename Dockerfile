FROM python:3.11-slim
ENV PYTHONUNBUFFERED=1 HF_HOME=/app/.hf PIP_NO_CACHE_DIR=1
WORKDIR /app
# CPU-only torch keeps the image small
RUN pip install torch==2.14.0 --index-url https://download.pytorch.org/whl/cpu
COPY requirements.txt .
RUN pip install -r requirements.txt
# bake the embedding model into the image so startup needs no download
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
COPY app app
COPY pipeline pipeline
COPY models models
COPY data/gazetteer data/gazetteer
COPY data/processed/main_results.json data/processed/main_results.json
COPY data/cache/osm data/cache/osm
RUN useradd -m -u 1000 user && chown -R user /app
USER user
EXPOSE 7860
# GEMINI_API_KEY is provided as a Space secret, never baked into the image
CMD ["uvicorn", "app.server:app", "--host", "0.0.0.0", "--port", "7860"]
