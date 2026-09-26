"""Bonus round: the global, multi-disaster dataset -> flood tweets on a world map.

  .venv/bin/python -m pipeline.build_bonus prepare            # group copies, mask, attach CrisisLexT26 truth (0 requests)
  .venv/bin/python -m pipeline.build_bonus sample             # Gemini v4 labels, 4,000 random non-CrisisLex groups (20 req.)
  .venv/bin/python -m pipeline.train                          # retrain with the sample (hazard head learns in-domain)
  .venv/bin/python -m pipeline.build_bonus evaluate           # local model vs CrisisLexT26 truth (0 requests)
  .venv/bin/python -m pipeline.build_bonus run --budget 68    # capped cascade over everything -> app data

Honest evaluation: ~21,400 bonus rows are CrisisLexT26 labeled tweets (event + informativeness). None of them are
used for training (the leakage guard in external.py drops them, and the Gemini sample is drawn only from the other
rows), so they are a held-out test set whose truth doesn't come from Gemini.
Proxy truth: flood = tweet from a flood event AND judged related. Caveat: typhoon tweets about flooding count as
"not flood" under this proxy, so precision against it is a lower bound.
"""
import argparse
import glob
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

from pipeline.clean import build, norm_key
from pipeline.context import build_context
from pipeline.gemini import Gemini, label_batch
from pipeline.taxonomy import EVENT_HAZARD

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data/raw/bonus_contestant.csv"
CLEAN = ROOT / "data/processed/bonus_clean.csv"
SAMPLE = ROOT / "data/labels/gemini_bonus_sample.csv"
OUT = ROOT / "data/processed/bonus_results.json"
EVAL = ROOT / "data/labels/eval_bonus.json"
T26 = ROOT / "data/raw/crisislext26/CrisisLexT26"
N_SAMPLE, SEED = 4000, 42


def t26_truth() -> pd.DataFrame:
    rows = []
    for f in glob.glob(str(T26 / "*/*-tweets_labeled.csv")):
        ev = os.path.basename(os.path.dirname(f))
        d = pd.read_csv(f, skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        rows.append(pd.DataFrame({"key": d["Tweet Text"].astype(str).map(norm_key), "t26_event": ev,
                                  "t26_info": d["Informativeness"]}))
    t = pd.concat(rows).drop_duplicates("key")
    t = t[t.t26_info.isin(["Related and informative", "Related - but not informative", "Not related"])]
    t["proxy_relevant"] = (t.t26_info != "Not related").astype(int)
    t["proxy_flood"] = ((t.t26_event.map(EVENT_HAZARD) == "FLOOD") & (t.proxy_relevant == 1)).astype(int)
    return t


def prepare():
    raw = pd.read_csv(RAW)
    df = build(raw)
    ctx = build_context(df["text"].tolist())
    print(f"mode={ctx.mode} countries={ctx.region.get('countries', [])[:6]}")
    df = df.merge(t26_truth(), on="key", how="left")
    df["masked"] = ""
    canon = df.is_canonical
    df.loc[canon, "masked"] = df.loc[canon, "text"].map(ctx.masker)
    cols = ["row_id", "group_id", "group_size", "is_canonical", "is_retweet", "text", "masked", "t26_event",
            "t26_info", "proxy_relevant", "proxy_flood"]
    df[cols].to_csv(CLEAN, index=False)
    c = df[canon]
    print(f"rows={len(df)} groups={int(canon.sum())} with CrisisLexT26 truth={int(c.t26_event.notna().sum())} "
          f"(flood={int(c.proxy_flood.sum())})")


def sample(max_calls: int):
    c = pd.read_csv(CLEAN, keep_default_na=False)
    c = c[(c.is_canonical.astype(str) == "True") & (c.t26_event == "")]
    pick = c.sample(N_SAMPLE, random_state=SEED).reset_index(drop=True)
    client = Gemini(budget=max_calls)
    if client.dry_run:
        raise SystemExit("GEMINI_API_KEY not set")
    batches = [pick.iloc[i:i + 200] for i in range(0, len(pick), 200)]

    def run(b):
        return b, label_batch(client, b["text"].tolist(), tag="bonus_sample:b200", v4=True)

    rows = []
    with ThreadPoolExecutor(max_workers=4) as ex:
        for b, res in ex.map(run, batches):
            for j, lab in res.items():
                rows.append({"group_id": int(b.iloc[j].group_id), "relevant": lab["relevant"],
                             "confidence": lab["confidence"] / 9, "category": lab["category"],
                             "urgency": lab["urgency"], "hazard": lab.get("hazard") or "",
                             "cc": lab.get("cc") or "", "locations": ";".join(lab["locations"])})
    out = pd.DataFrame(rows)
    out.to_csv(SAMPLE, index=False)
    r = out[out.relevant == 1]
    print(f"labeled {len(out)}/{len(pick)} | calls this run: {client.spent} | ledger total: {Gemini.ledger_total()}")
    print(f"relevant {len(r)} | hazards {r.hazard.value_counts().to_dict()} | countries {r.cc.value_counts().head(8).to_dict()}")


def _scores(y, p, thr=0.5):
    yhat = (p >= thr).astype(int)
    return {"n": int(len(y)), "positives": int(y.sum()), "auc": round(float(roc_auc_score(y, p)), 3),
            "precision": round(float(precision_score(y, yhat)), 3), "recall": round(float(recall_score(y, yhat)), 3),
            "f1": round(float(f1_score(y, yhat)), 3)}


def evaluate():
    from pipeline.model import LocalModel, featurize
    c = pd.read_csv(CLEAN, keep_default_na=False)
    held = c[(c.is_canonical.astype(str) == "True") & (c.t26_event != "")].reset_index(drop=True)
    if SAMPLE.exists():  # by construction the sample has no CrisisLex rows; assert it
        assert not set(pd.read_csv(SAMPLE).group_id) & set(held.group_id), "sample overlaps the held-out set"
    pred = LocalModel.load().predict(featurize(held["masked"].tolist(), "bonus_heldout"))
    y_flood = held.proxy_flood.astype(float).astype(int).values
    y_rel = held.proxy_relevant.astype(float).astype(int).values
    res = {"local_flood": _scores(y_flood, pred["p_flood"]), "local_relevant": _scores(y_rel, pred["p_relevant"])}
    per_event = held.assign(p=pred["p_flood"]).groupby("t26_event").apply(
        lambda g: round(float((g.p >= 0.5).mean()), 3), include_groups=False)
    res["share_flagged_flood_per_event"] = per_event.sort_values(ascending=False).to_dict()
    print(json.dumps(res, indent=1))
    prev = json.loads(EVAL.read_text()) if EVAL.exists() else {}
    EVAL.write_text(json.dumps({**prev, "local": res}, indent=1))


def run(budget: int, replay: bool = False):
    """replay=True rebuilds the output from cached Gemini answers only (0 requests)."""
    from app.server import to_records
    from pipeline.cascade import process
    raw = pd.read_csv(RAW)
    pre = pd.read_csv(SAMPLE, keep_default_na=False) if SAMPLE.exists() else None
    client = Gemini(budget=budget)
    client.cache_only = replay
    df, stats = process(raw, budget=budget, batch_size=200, gemini=client, precomputed=pre,
                        geocode_limit=300, progress=lambda m, f: print(f"  {m}", flush=True))
    stats = json.loads(json.dumps(stats, default=str))
    OUT.write_text(json.dumps({"name": "Global disasters (bonus dataset)", "stats": stats, "records": to_records(df)},
                              separators=(",", ":")))
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e6:.1f} MB) | calls {stats.get('gemini_calls_spent')} | "
          f"flood tweets (unique) {stats.get('flood_unique')} | decided_by {stats.get('decided_by')}")
    # final system vs CrisisLexT26 truth, on held-out labeled groups
    c = pd.read_csv(CLEAN, keep_default_na=False)
    held = c[(c.is_canonical.astype(str) == "True") & (c.t26_event != "")][["group_id", "proxy_flood"]]
    final = df[df.is_canonical][["group_id", "relevant", "hazard", "p_flood", "decided_by"]].merge(held, on="group_id")
    yhat = ((final.relevant == 1) & (final.hazard == "FLOOD")).astype(int)
    y = final.proxy_flood.astype(float).astype(int)
    sys_scores = {"n": int(len(y)), "positives": int(y.sum()),
                  "precision": round(float(precision_score(y, yhat)), 3), "recall": round(float(recall_score(y, yhat)), 3),
                  "f1": round(float(f1_score(y, yhat)), 3), "gemini_reviewed_share": round(float((final.decided_by == "gemini").mean()), 3)}
    print("final system (flood) vs CrisisLexT26:", sys_scores)
    prev = json.loads(EVAL.read_text()) if EVAL.exists() else {}
    EVAL.write_text(json.dumps({**prev, "system": sys_scores}, indent=1))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["prepare", "sample", "evaluate", "run"])
    ap.add_argument("--max-calls", type=int, default=22)
    ap.add_argument("--budget", type=int, default=68)
    ap.add_argument("--replay", action="store_true", help="rebuild from cached Gemini answers only (0 requests)")
    a = ap.parse_args()
    {"prepare": prepare, "sample": lambda: sample(a.max_calls), "evaluate": evaluate,
     "run": lambda: run(a.budget, a.replay)}[a.cmd]()
