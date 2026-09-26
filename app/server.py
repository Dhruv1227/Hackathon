"""Living Flood Map web server.

  .venv/bin/uvicorn app.server:app --port 7860

Serves the static map UI, the precomputed main dataset, CSV uploads (processed in a background
thread with a per-upload Gemini call cap), and the one-call "Summarize this view".
"""
import gzip
import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from app.packing import compress_payload, summary_rows
from pipeline.cascade import process
from pipeline.gemini import Gemini
from pipeline.summary import summarize
from pipeline.taxonomy import CATEGORIES, HAZARDS
from pipeline.world import load_countries

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
PROCESSED = ROOT / "data/processed"
UPLOAD_BUDGET = int(os.getenv("UPLOAD_CALL_BUDGET", "20"))
BATCH_SIZE = int(os.getenv("GEMINI_BATCH_SIZE", "200"))  # chosen on the gold set: 200/call lost no accuracy
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "50000"))  # the Render image sets 20000
SUMMARY_CALLS_PER_DATASET = int(os.getenv("SUMMARY_CALLS_PER_DATASET", "20"))

app = FastAPI(title="Living Flood Map")
# id -> {"name", "rows", "stats", "gz": gzipped API payload, "summary": slim rows or None, "summary_path"}
DATASETS: dict[str, dict] = {}
JOBS: dict[str, dict] = {}
_model_lock = threading.Lock()


def to_records(df: pd.DataFrame) -> list[dict]:
    recs = []
    for r in df.itertuples():
        recs.append({
            "id": int(r.row_id), "g": int(r.group_id), "n": int(r.group_size), "rt": bool(r.is_retweet),
            "canon": bool(r.is_canonical), "text": r.text, "rel": int(r.relevant),
            "p": round(float(r.p_relevant), 3), "conf": round(float(r.confidence), 2),
            "cat": r.category if isinstance(r.category, str) else "", "urg": int(r.urgency),
            "hz": (getattr(r, "hazard", "") or "") if isinstance(getattr(r, "hazard", ""), str) else "",
            "cc": getattr(r, "cc", None) if isinstance(getattr(r, "cc", None), str) else None,
            "by": r.decided_by,
            "places": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"], "prec": p.get("precision", "point"),
                        "src": p.get("source", ""), "cc": p.get("cc")}
                       for p in (r.places if isinstance(r.places, list) else [])],
            "ts": (r.timestamp.isoformat() if hasattr(r, "timestamp") and pd.notna(r.timestamp) else None),
        })
    return recs


def register_payload(ds_id: str, payload: dict):
    """Keep a dataset compactly: gzipped payload for /api/data, slim rows for summaries; no record objects."""
    DATASETS[ds_id] = {"name": payload["name"], "rows": len(payload["records"]), "stats": payload["stats"],
                       "gz": compress_payload(payload), "summary": summary_rows(payload["records"]),
                       "summary_path": None, "summary_calls": 0}


def register(name: str, df: pd.DataFrame, stats: dict) -> str:
    ds_id = uuid.uuid4().hex[:10]
    register_payload(ds_id, {"name": name, "stats": _jsonable(stats), "records": to_records(df)})
    return ds_id


def load_builtin():
    for ds_id, stem in (("main", "main_results"), ("bonus", "bonus_results")):
        meta, gz = PROCESSED / f"{stem}.meta.json", PROCESSED / f"{stem}.json.gz"
        if meta.exists() and gz.exists():  # packed (deploy): never parse the big payload
            m = json.loads(meta.read_text())
            DATASETS[ds_id] = {"name": m["name"], "rows": m["rows"], "stats": m["stats"], "gz": gz.read_bytes(),
                               "summary": None, "summary_path": PROCESSED / f"{stem}.summary.json.gz",
                               "summary_calls": 0}
        elif (PROCESSED / f"{stem}.json").exists():  # local development: plain JSON
            register_payload(ds_id, json.loads((PROCESSED / f"{stem}.json").read_text()))


def summary_index(ds: dict) -> list[list]:
    if ds["summary"] is None:
        with gzip.open(ds["summary_path"], "rt", encoding="utf-8") as f:
            ds["summary"] = json.load(f)
    return ds["summary"]


load_builtin()


# ---------------------------------------------------------------- API

@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "hazards": HAZARDS, "gemini": not Gemini().dry_run,
            "upload_budget": UPLOAD_BUDGET, "batch_size": BATCH_SIZE,
            "countries": {cc: {"name": c["name"], "lat": c["lat"], "lon": c["lon"]} for cc, c in load_countries().items()},
            "datasets": [{"id": k, "name": v["name"], "rows": v["rows"]} for k, v in DATASETS.items()]}


@app.get("/api/health")
def health():
    return {"ok": True, "datasets": list(DATASETS)}


@app.get("/api/data/{ds_id}")
def data(ds_id: str):
    ds = DATASETS.get(ds_id) or _404()
    return Response(ds["gz"], media_type="application/json", headers={"Content-Encoding": "gzip"})


@app.post("/api/upload")
async def upload(file: UploadFile = File(...), budget: int = Form(UPLOAD_BUDGET)):
    budget = max(0, min(budget, UPLOAD_BUDGET))
    raw_bytes = await file.read()
    try:
        raw = pd.read_csv(io.BytesIO(raw_bytes), encoding_errors="replace", on_bad_lines="skip")
    except Exception as e:
        raise HTTPException(400, f"Could not read CSV: {e}")
    if raw.empty:
        raise HTTPException(400, "CSV has no rows")
    if len(raw) > MAX_UPLOAD_ROWS:
        raw = raw.head(MAX_UPLOAD_ROWS)
    job_id = uuid.uuid4().hex[:10]
    JOBS[job_id] = {"status": "running", "msg": "Queued", "frac": 0.0, "dataset": None, "error": None}

    def work():
        def progress(msg, frac):
            JOBS[job_id].update(msg=msg, frac=frac)
        try:
            with _model_lock:
                df, stats = process(raw, budget=budget, batch_size=BATCH_SIZE, gemini=Gemini(budget=budget),
                                    progress=progress)
            JOBS[job_id]["dataset"] = register(file.filename or "upload.csv", df, stats)
            JOBS[job_id]["status"] = "done"
        except Exception as e:
            JOBS[job_id].update(status="error", error=f"{type(e).__name__}: {e}")

    threading.Thread(target=work, daemon=True).start()
    return {"job": job_id}


@app.get("/api/jobs/{job_id}")
def job(job_id: str):
    return JOBS.get(job_id) or _404()


class SummaryReq(BaseModel):
    groups: list[int]
    filters: str = "all relevant tweets"


@app.post("/api/summary/{ds_id}")
def summary(ds_id: str, req: SummaryReq):
    ds = DATASETS.get(ds_id) or _404()
    if ds["summary_calls"] >= SUMMARY_CALLS_PER_DATASET:
        raise HTTPException(429, "Summary call limit reached for this dataset")
    want = set(req.groups)
    view = pd.DataFrame([{"row_id": r[0], "text": r[2], "group_size": r[3], "urgency": r[4], "category": r[5],
                          "confidence": r[6], "hazard": r[7], "cc": r[8], "places": [{"name": n} for n in r[9]]}
                         for r in summary_index(ds) if r[1] in want])
    res = summarize(view, req.filters, Gemini(budget=1, allow_reserve=True))
    ds["summary_calls"] += res["calls"]
    return res


def _404():
    raise HTTPException(404, "not found")


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


@app.get("/")
def index():
    # version-stamp the static files so browsers never run a stale app.js after a redeploy
    html = (STATIC / "index.html").read_text()
    for name in ("app.js", "style.css"):
        html = html.replace(f"/static/{name}", f"/static/{name}?v={int((STATIC / name).stat().st_mtime)}")
    return HTMLResponse(html, headers={"Cache-Control": "no-cache"})


app.mount("/static", StaticFiles(directory=STATIC), name="static")
