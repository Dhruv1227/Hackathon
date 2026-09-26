"""Living Flood Map web server.

  .venv/bin/uvicorn app.server:app --port 7860

Serves the static map UI, the precomputed main dataset, CSV uploads (processed in a background
thread with a per-upload Gemini call cap), and the one-call "Summarize this view".
"""
import io
import json
import os
import threading
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from pipeline.cascade import process
from pipeline.gemini import Gemini
from pipeline.summary import summarize
from pipeline.taxonomy import CATEGORIES

ROOT = Path(__file__).resolve().parent.parent
STATIC = Path(__file__).resolve().parent / "static"
MAIN_RESULTS = ROOT / "data/processed/main_results.json"
UPLOAD_BUDGET = int(os.getenv("UPLOAD_CALL_BUDGET", "20"))
BATCH_SIZE = int(os.getenv("GEMINI_BATCH_SIZE", "200"))  # chosen on the gold set: 200/call lost no accuracy
MAX_UPLOAD_ROWS = int(os.getenv("MAX_UPLOAD_ROWS", "50000"))
SUMMARY_CALLS_PER_DATASET = int(os.getenv("SUMMARY_CALLS_PER_DATASET", "20"))

app = FastAPI(title="Living Flood Map")
DATASETS: dict[str, dict] = {}   # id -> {"name", "records", "stats", "df"}
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
            "by": r.decided_by,
            "places": [{"name": p["name"], "lat": p["lat"], "lon": p["lon"], "prec": p.get("precision", "point"),
                        "src": p.get("source", "")} for p in (r.places if isinstance(r.places, list) else [])],
            "ts": (r.timestamp.isoformat() if hasattr(r, "timestamp") and pd.notna(r.timestamp) else None),
        })
    return recs


def register(name: str, df: pd.DataFrame, stats: dict) -> str:
    ds_id = uuid.uuid4().hex[:10] if name != "main" else "main"
    DATASETS[ds_id] = {"name": name, "df": df, "stats": stats, "records": to_records(df), "summary_calls": 0}
    return ds_id


def load_main():
    if MAIN_RESULTS.exists():
        d = json.loads(MAIN_RESULTS.read_text())
        DATASETS["main"] = {"name": d["name"], "records": d["records"], "stats": d["stats"], "df": None,
                            "summary_calls": 0}


load_main()


# ---------------------------------------------------------------- API

@app.get("/api/meta")
def meta():
    return {"categories": CATEGORIES, "gemini": not Gemini().dry_run, "upload_budget": UPLOAD_BUDGET,
            "batch_size": BATCH_SIZE,
            "datasets": [{"id": k, "name": v["name"], "rows": len(v["records"])} for k, v in DATASETS.items()]}


@app.get("/api/data/{ds_id}")
def data(ds_id: str):
    ds = DATASETS.get(ds_id) or _404()
    return JSONResponse({"name": ds["name"], "stats": _jsonable(ds["stats"]), "records": ds["records"]})


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
    rows = [r for r in ds["records"] if r["canon"] and r["rel"] == 1 and r["g"] in want]
    view = pd.DataFrame([{"row_id": r["id"], "text": r["text"], "group_size": r["n"], "urgency": r["urg"],
                          "category": r["cat"] or "OTHER", "confidence": r["conf"], "places": r["places"]}
                         for r in rows])
    res = summarize(view, req.filters, Gemini(budget=1, allow_reserve=True))
    ds["summary_calls"] += res["calls"]
    return res


def _404():
    raise HTTPException(404, "not found")


def _jsonable(o):
    return json.loads(json.dumps(o, default=lambda x: x.item() if isinstance(x, np.generic) else str(x)))


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
