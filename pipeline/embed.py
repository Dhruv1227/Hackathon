"""Local sentence embeddings (no Gemini calls). Cached on disk by text hash.

Two interchangeable backends for the same model (all-MiniLM-L6-v2, mean pooling, L2-normalised):
  onnx  ONNX Runtime + the model's official ONNX export: ~150 MB RAM, no PyTorch (used for deployment)
  st    sentence-transformers / PyTorch (what the model was first trained with)
Chosen by EMBED_BACKEND; default onnx when onnxruntime is installed. The two agree to ~1e-6 cosine.
"""
import hashlib
import os
from functools import lru_cache
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
MAX_TOKENS = 256  # the model's max_seq_length
CACHE = ROOT / "data/cache/emb"


def backend() -> str:
    b = os.getenv("EMBED_BACKEND", "").lower()
    if b in ("onnx", "st"):
        return b
    try:
        import onnxruntime  # noqa: F401
        return "onnx"
    except ImportError:
        return "st"


@lru_cache(maxsize=1)
def model():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME, device="cpu")


@lru_cache(maxsize=1)
def onnx_model():
    import onnxruntime as ort
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer
    tok = Tokenizer.from_file(hf_hub_download(MODEL_NAME, "tokenizer.json"))
    tok.enable_truncation(max_length=MAX_TOKENS)
    tok.enable_padding(pad_id=0, pad_token="[PAD]")
    opts = ort.SessionOptions()
    opts.intra_op_num_threads = int(os.getenv("EMBED_THREADS", "0"))
    if os.getenv("EMBED_PREPACK") != "1":  # pre-packing copies the weights (~90 MB) for a little speed; skip it
        opts.add_session_config_entry("session.disable_prepacking", "1")
    sess = ort.InferenceSession(hf_hub_download(MODEL_NAME, "onnx/model.onnx"), sess_options=opts,
                                providers=["CPUExecutionProvider"])
    return tok, sess


def _encode_onnx(texts: list[str], batch_size: int) -> np.ndarray:
    tok, sess = onnx_model()
    names = {i.name for i in sess.get_inputs()}
    out = []
    for i in range(0, len(texts), batch_size):
        enc = tok.encode_batch(texts[i:i + batch_size])
        ids = np.array([e.ids for e in enc], dtype=np.int64)
        mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
        feeds = {"input_ids": ids, "attention_mask": mask}
        if "token_type_ids" in names:
            feeds["token_type_ids"] = np.zeros_like(ids)
        hidden = sess.run(None, feeds)[0]
        m = mask[..., None].astype(np.float32)
        emb = (hidden * m).sum(1) / np.clip(m.sum(1), 1e-9, None)  # mean pooling over real tokens
        emb /= np.clip(np.linalg.norm(emb, axis=1, keepdims=True), 1e-12, None)
        out.append(emb.astype(np.float32))
    return np.vstack(out) if out else np.zeros((0, 384), np.float32)


def embed(texts: list[str], cache_name: str | None = None, batch_size: int | None = None) -> np.ndarray:
    if cache_name:
        h = hashlib.sha256("\n".join(texts).encode()).hexdigest()[:16]
        path = CACHE / f"{cache_name}-{h}.npy"
        if path.exists():
            return np.load(path)
    if backend() == "onnx":
        # small batches: batch 64 needs ~380 MB of activations, batch 8 ~180 MB, same output, slightly faster
        vecs = _encode_onnx(texts, batch_size or int(os.getenv("EMBED_BATCH", "8")))
    else:
        vecs = model().encode(texts, batch_size=batch_size or 64, normalize_embeddings=True, show_progress_bar=False)
    vecs = vecs.astype(np.float32)
    if cache_name:
        CACHE.mkdir(parents=True, exist_ok=True)
        np.save(path, vecs)
    return vecs
