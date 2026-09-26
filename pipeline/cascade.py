"""End-to-end processing of an uploaded CSV with a per-upload Gemini call budget.

1. Detect text/timestamp columns, clean, group copies (retweets join originals).
2. Local model scores every canonical tweet (relevance, category, urgency) + gazetteer places.
3. Budget: cap = `budget` calls x `batch_size` tweets. Queue for Gemini, in order:
     a. uncertain band (local p_relevant in [LO, HI]), most uncertain first
     b. relevant tweets that seem to name a place the gazetteer couldn't match
     c. everything else, most uncertain first (small files get ~full Gemini coverage)
   A 2,000-tweet file is almost fully covered; a 20,000-tweet file still spends <= budget calls.
4. Gemini labels override local ones; Gemini places are resolved via gazetteer, then geocoder.
5. Labels propagate from the canonical tweet to every copy in its group.
"""
import math
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pandas as pd

from pipeline.clean import build
from pipeline.context import build_context
from pipeline.gemini import BudgetExceeded, Gemini, label_batch
from pipeline.gazetteer import TOO_BROAD
from pipeline.geocode import geocode
from pipeline.model import LocalModel, featurize

LO, HI = 0.15, 0.85
TEXT_COLS = ["tweet", "text", "tweet_text", "full_text", "content", "message", "body", "tweet text"]
TIME_COLS = ["created_at", "timestamp", "date", "datetime", "time", "tweet_time", "posted_at", "created"]
PLACE_HINT_RE = re.compile(
    r"\b(?:in|at|near|on|by|across|along|off)\s+(?:the\s+)?([A-Z][\w'’]+(?:\s+[A-Z][\w'’]+){0,3})"
    r"|\b(\d+\s*(?:st|nd|rd|th)?\s+(?:St|Street|Ave|Avenue|Rd|Road|Dr|Drive|Blvd|Trail|Way|Cres)\b)"
    r"|\b([A-Z][\w'’]+\s+(?:Trail|Drive|Road|Bridge|Crossing|Park|Street|Avenue|Creek|Lake|Reserve|Nation))\b")


def detect_columns(df: pd.DataFrame):
    lower = {c.lower().strip(): c for c in df.columns}
    text_col = next((lower[c] for c in TEXT_COLS if c in lower), None)
    if text_col is None:  # longest average string column
        obj = [c for c in df.columns if df[c].dtype == object]
        text_col = max(obj, key=lambda c: df[c].astype(str).str.len().mean())
    time_col = next((lower[c] for c in TIME_COLS if c in lower), None)
    return text_col, time_col


def place_hint(text: str, found: list[dict]) -> bool:
    """True if the tweet seems to name a specific place the gazetteer didn't resolve."""
    surfaces = " ".join(p["surface"].lower() for p in found)
    for m in PLACE_HINT_RE.finditer(text):
        s = next(g for g in m.groups() if g)
        if s.lower() not in surfaces and s.split()[0].lower() not in ("i", "the", "my", "our", "rt"):
            return True
    return False


def plan_budget(local: pd.DataFrame, budget: int, batch_size: int) -> list[int]:
    """Canonical-row indices to send to Gemini, in priority order, within the budget. The map is about floods,
    so the ranking uses p_flood = P(relevant) x P(hazard is flood):
      1. uncertain band: p_flood in [LO, HI], most uncertain first
      2. likely floods (Gemini adds category, urgency, places, country): ones naming an unmatched place first
      3. everything else, most uncertain about relevance first"""
    reserve = 1 if budget >= 10 else 0  # one spare call for retrying dropped lines
    cap = max(0, budget - reserve) * batch_size
    p = local["p_flood"]
    band = local[(p >= LO) & (p <= HI)].sort_values("flood_uncertainty", ascending=False).index.tolist()
    likely = local[p > HI].assign(h=lambda d: ~d["place_hint"]).sort_values(["h", "p_flood"], ascending=[True, False])
    chosen = set(band) | set(likely.index)
    rest = local[~local.index.isin(chosen)].sort_values("uncertainty", ascending=False).index.tolist()
    return (band + likely.index.tolist() + rest)[:cap]


def resolve_places(names: list[str], ctx, geocode_budget: list[int], cc: str | None = None) -> list[dict]:
    """Resolve Gemini's place names: gazetteer first (with Gemini's country as a hint), then the geocoder."""
    out, seen = [], set()
    single = ctx.mode == "single" and ctx.region and ctx.region.get("bbox")
    for n in names:
        if n.lower().strip() in TOO_BROAD:
            continue
        p = ctx.gazetteer.resolve(n, cc)
        if p is None and geocode_budget[0] > 0:
            geocode_budget[0] -= 1
            country = (cc or (ctx.region or {}).get("country") or "").lower() or None
            p = geocode(n, bbox=ctx.region["bbox"] if single else None, country=country)
        if p and p["name"] not in seen:
            seen.add(p["name"])
            out.append({**p, "surface": n})
    return out


def process(raw: pd.DataFrame, budget: int = 20, batch_size: int = 200, gemini: Gemini | None = None,
            model: LocalModel | None = None, progress=lambda msg, frac: None, precomputed: pd.DataFrame | None = None,
            geocode_limit: int = 80, concurrency: int = 4,
            default_hazard: str | None = None) -> tuple[pd.DataFrame, dict]:
    text_col, time_col = detect_columns(raw)
    progress("Cleaning and grouping copies", 0.02)
    df = build(raw.rename(columns={text_col: "__text"}), text_col="__text")
    if time_col:
        df["timestamp"] = pd.to_datetime(raw.loc[df["row_id"], time_col].values, errors="coerce", utc=True)
    canon = df[df.is_canonical].copy()

    progress("Detecting event region and building gazetteer", 0.08)
    ctx = build_context(df["text"].tolist())
    canon["masked"] = canon["text"].map(ctx.masker)
    canon["places"] = canon["text"].map(ctx.gazetteer.extract)
    canon["place_hint"] = [place_hint(t, f) for t, f in zip(canon["text"], canon["places"])]

    progress(f"Local model scoring {len(canon)} unique tweets", 0.15)
    model = model or LocalModel.load()
    pred = model.predict(featurize(canon["masked"].tolist()))
    for k in ("p_relevant", "category", "urgency", "uncertainty", "cat_conf", "hazard", "p_flood",
              "flood_uncertainty"):
        canon[k] = pred[k]
    canon["relevant"] = (canon["p_relevant"] >= 0.5).astype(int)
    canon["confidence"] = np.abs(canon["p_relevant"] - 0.5) * 2
    canon["decided_by"] = "local"
    canon["gemini_places"] = [[] for _ in range(len(canon))]
    canon["gemini_cc"] = None

    # labels already produced offline (main dataset, bonus sample) are reused without spending calls
    if precomputed is not None and len(precomputed):
        pc = precomputed.drop_duplicates("group_id").set_index("group_id")
        hit = canon["group_id"].isin(pc.index)
        for idx in canon.index[hit]:
            r = pc.loc[canon.at[idx, "group_id"]]
            hz = r.get("hazard") if isinstance(r.get("hazard"), str) and r.get("hazard") else default_hazard
            cc = r.get("cc") if isinstance(r.get("cc"), str) and len(str(r.get("cc"))) == 2 else None
            _apply_gemini(canon, idx, r["relevant"], r["confidence"], r["category"], r["urgency"],
                          [s for s in str(r["locations"]).split(";") if s and s != "nan"], hz, cc)

    gemini = gemini or Gemini(budget=budget)
    todo = [i for i in plan_budget(canon[canon.decided_by == "local"], budget, batch_size)]
    batches = [todo[i:i + batch_size] for i in range(0, len(todo), batch_size)]
    stats = {"text_col": text_col, "time_col": time_col, "rows": len(df), "unique": len(canon),
             "gemini_planned_calls": len(batches), "gemini_dry_run": gemini.dry_run,
             "region": ctx.region, "budget": budget, "batch_size": batch_size}

    if batches and not gemini.dry_run:
        top = ", ".join(n for n, _ in (ctx.region or {}).get("top", [])[:3]) if ctx.mode == "single" else ""
        event_hint = f"an event affecting {top}" if top else None
        done = [0]

        def run(b):
            texts = canon.loc[b, "text"].tolist()
            try:
                res = label_batch(gemini, texts, tag=f"upload:{len(texts)}", event_hint=event_hint, v4=True)
            except BudgetExceeded as e:  # quota floor reached: these tweets keep their local labels
                stats["gemini_stopped"] = str(e)
                res = {}
            except Exception as e:  # proxy error: never fail the whole upload over one batch
                stats.setdefault("gemini_errors", []).append(f"{type(e).__name__}: {str(e)[:120]}")
                res = {}
            done[0] += 1
            progress(f"Gemini labeling: batch {done[0]}/{len(batches)}", 0.25 + 0.6 * done[0] / len(batches))
            return b, res

        with ThreadPoolExecutor(max_workers=concurrency) as ex:
            for b, res in ex.map(run, batches):
                for j, lab in res.items():
                    _apply_gemini(canon, b[j], lab["relevant"], lab["confidence"] / 9, lab["category"],
                                  lab["urgency"], lab["locations"], lab.get("hazard"), lab.get("cc"))
    stats["gemini_calls_spent"] = gemini.spent

    progress("Resolving locations", 0.88)
    geo_budget = [geocode_limit]
    merged = []
    for idx in canon.index:
        found = list(canon.at[idx, "places"])
        names = {p["name"] for p in found}
        extra = [n for n in canon.at[idx, "gemini_places"] if n not in names]
        if extra and canon.at[idx, "relevant"] == 1:
            found += [p for p in resolve_places(extra, ctx, geo_budget, canon.at[idx, "gemini_cc"])
                      if p["name"] not in names]
        merged.append(found)
    canon["places"] = merged
    canon["cc"] = [tweet_country(g, ps, ctx) for g, ps in zip(canon["gemini_cc"], canon["places"])]

    keep = ["group_id", "relevant", "p_relevant", "p_flood", "confidence", "category", "urgency", "hazard", "cc",
            "decided_by", "places", "place_hint"]
    out = df.drop(columns=[c for c in ("key", "truncated") if c in df]).merge(canon[keep], on="group_id", how="left")
    out.loc[out.relevant == 0, ["category", "hazard"]] = ""
    out.loc[out.relevant == 0, ["urgency"]] = 0
    stats["relevant_unique"] = int(canon["relevant"].sum())
    stats["flood_unique"] = int(((canon["relevant"] == 1) & (canon["hazard"] == "FLOOD")).sum())
    stats["mode"] = ctx.mode
    stats["decided_by"] = canon["decided_by"].value_counts().to_dict()
    progress("Done", 1.0)
    return out, stats


def _apply_gemini(canon, idx, rel, conf, cat, urg, locs, hazard=None, cc=None):
    canon.at[idx, "relevant"] = int(rel)
    canon.at[idx, "confidence"] = float(conf)
    if int(rel) == 1:
        canon.at[idx, "category"] = cat or canon.at[idx, "category"]
        canon.at[idx, "urgency"] = int(urg)
        if hazard:
            canon.at[idx, "hazard"] = hazard
        canon.at[idx, "gemini_cc"] = cc
    canon.at[idx, "decided_by"] = "gemini"
    canon.at[idx, "gemini_places"] = list(locs)


def tweet_country(gemini_cc, places: list[dict], ctx) -> str | None:
    """Gemini's country if it gave one; else the country most of the tweet's places are in (specific places
    count double); else the event's country for single-event datasets."""
    if isinstance(gemini_cc, str) and len(gemini_cc) == 2:
        return gemini_cc.upper()
    votes = Counter()
    for p in places:
        if p.get("cc"):
            votes[p["cc"]] += 1 if p.get("precision") == "region" else 2
    if votes:
        return votes.most_common(1)[0][0]
    return (ctx.region or {}).get("country") if ctx.mode == "single" else None
