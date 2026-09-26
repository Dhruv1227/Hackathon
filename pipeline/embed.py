"""Local sentence embeddings (no Gemini calls). Cached on disk by text hash."""
import hashlib
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
CACHE = ROOT / "data/cache/emb"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


def embed(texts: list[str], cache_name: str | None = None, batch_size: int = 128) -> np.ndarray:
    if cache_name:
        h = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        path = CACHE / f"{cache_name}-{h}.npy"
        if path.exists():
            return np.load(path)
    vecs = model().encode(texts, batch_size=batch_size, normalize_embeddings=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    if cache_name:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs
