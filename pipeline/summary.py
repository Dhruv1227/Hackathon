"""One-call situation summary of the current view.

Representative tweets are picked locally (most shared, most urgent, spread across categories),
then summarised in a single Gemini call. Only triggered by the "Summarize this view" button.
"""
import math

import pandas as pd

from pipeline.gemini import Gemini
from pipeline.taxonomy import CATEGORIES

N_PICK = 150


def pick_representative(view: pd.DataFrame, n: int = N_PICK) -> pd.DataFrame:
    """view: one row per unique tweet (canonical) with group_size, urgency, category, confidence."""
    v = view.copy()
    v["score"] = (v["group_size"].map(lambda s: math.log2(1 + s)) + 1.5 * v["urgency"].astype(float)
                  + v["confidence"].astype(float) + v["places"].map(lambda p: 0.5 if len(p) else 0))
    v = v.sort_values("score", ascending=False)
    cats = v["category"].value_counts()
    # every category gets at least a few slots, the rest proportional to its size
    quota = {c: max(3, round(n * k / len(v))) for c, k in cats.items()}
    picked = pd.concat([v[v.category == c].head(q) for c, q in quota.items()])
    if len(picked) < n:
        picked = pd.concat([picked, v[~v.index.isin(picked.index)].head(n - len(picked))])
    return picked.sort_values("score", ascending=False).head(n)


def summary_prompt(picked: pd.DataFrame, filters_desc: str, total: int) -> str:
    from pipeline.world import country_name
    lines = []
    for i, r in enumerate(picked.itertuples()):
        places = ",".join(p["name"] for p in r.places[:3]) or "-"
        cc = getattr(r, "cc", None)
        cc = cc if isinstance(cc, str) and cc else None  # pandas turns a missing country into NaN
        where = f"{places}; {country_name(cc)}" if cc else places
        hz = getattr(r, "hazard", "")
        hz = f"{hz}, " if isinstance(hz, str) and hz else ""
        lines.append(f"[{i}] ({hz}{r.category}, urgency {r.urgency}, shared {r.group_size}x, where: {where}) {r.text}")
    cats = ", ".join(f"{k}={v}" for k, v in CATEGORIES.items())
    return f"""You are briefing flood emergency coordinators and First Nations community leaders.
Below are {len(picked)} representative tweets chosen from {total} relevant tweets matching: {filters_desc}.
Categories: {cats}

Write a situation summary in Markdown, at most 250 words, with these sections:
**Overview** (2-3 sentences: what is happening, where, how severe)
**Most affected places** (bullets: place and country, what is reported there; group by country if several)
**Urgent needs & safety issues** (bullets; say "none reported" if none)
**Infrastructure & services** (bullets: roads, bridges, power, water)
**Response & aid** (bullets)
Cite supporting tweets inline as [n]. Report only what the tweets say; flag unverified claims.

Tweets:
""" + "\n".join(lines)


def summarize(view: pd.DataFrame, filters_desc: str, gemini: Gemini | None = None) -> dict:
    if view.empty:
        return {"markdown": "_No relevant tweets in this view._", "cited": [], "calls": 0}
    picked = pick_representative(view)
    gemini = gemini or Gemini()
    if gemini.dry_run:
        top = picked.head(8)
        md = ("**Gemini key not configured — showing the top representative tweets instead.**\n\n" +
              "\n".join(f"- ({r.category}, urgency {r.urgency}) {r.text}" for r in top.itertuples()))
        return {"markdown": md, "cited": top["row_id"].tolist(), "calls": 0}
    before = gemini.spent
    md = gemini.generate(summary_prompt(picked, filters_desc, len(view)), tag="summary", temperature=0.2)
    return {"markdown": md, "cited": picked["row_id"].tolist(), "calls": gemini.spent - before}
