"""Precompute the main dataset for the app (no Gemini calls here: reuses offline Gemini labels).

  .venv/bin/python -m pipeline.build_main
"""
import json
from pathlib import Path

import pandas as pd

from app.server import to_records
from pipeline.cascade import process
from pipeline.gemini import Gemini

ROOT = Path(__file__).resolve().parent.parent
GEMINI_LABELS = ROOT / "data/labels/gemini_labels.csv"
OUT = ROOT / "data/processed/main_results.json"


def main():
    raw = pd.read_csv(ROOT / "data/raw/main_contestant.csv")
    # train labels + the tune/gold labels already paid for during evaluation (gold at the chosen 200/call)
    parts = [ROOT / "data/labels" / f for f in ("gemini_labels.csv", "gemini_gold_b200.csv", "gemini_tune.csv")]
    pre = pd.concat([pd.read_csv(f, keep_default_na=False) for f in parts if f.exists()]) if any(
        f.exists() for f in parts) else None
    g = Gemini(budget=0)
    g.dry_run = True  # never spend calls while building the shipped dataset
    # the main dataset is one flood event, and its Gemini labels (frozen v3 prompt) predate the hazard field
    df, stats = process(raw, budget=0, gemini=g, precomputed=pre, default_hazard="FLOOD", geocode_limit=300,
                        progress=lambda m, f: print(f"  {m}"))
    stats = json.loads(json.dumps(stats, default=str))
    OUT.write_text(json.dumps({"name": "2013 Alberta floods (provided dataset)", "stats": stats,
                               "records": to_records(df)}, separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) relevant_unique={stats['relevant_unique']} "
          f"decided_by={stats['decided_by']}")


if __name__ == "__main__":
    main()
