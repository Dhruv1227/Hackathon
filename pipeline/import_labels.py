"""Validate the hand-labeled CSV and store it as data/labels/hand_labels.csv.

  .venv/bin/python -m pipeline.import_labels path/to/to_label.csv
"""
import sys
from pathlib import Path

import pandas as pd

from pipeline.taxonomy import CATEGORY_CODES

CAT_FIX = {"NEED": "NEEDS", "OTHERS": "OTHER", "INFRASTRUCTURE": "INFRA", "EVACUATION": "EVAC", "ADVICE": "ADVISORY"}
PROVINCE_ABBR = {"AB": "Alberta", "BC": "British Columbia", "SK": "Saskatchewan", "MB": "Manitoba", "ON": "Ontario"}


def norm_locations(s: str) -> str:
    """'High River,AB' / 'Calgary, AB' / 'Bow River; Calgary' -> 'High River;Alberta' etc."""
    out = []
    for part in pd.Series(s.replace(",", ";").split(";")).str.strip():
        if not part:
            continue
        part = PROVINCE_ABBR.get(part.upper(), part)
        if part.lower() not in (o.lower() for o in out):
            out.append(part)
    return ";".join(out)

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "data/labels/hand_labels.csv"


def main(path):
    d = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    d.columns = [c.strip().lower() for c in d.columns]
    need = {"group_id", "split", "relevant"}
    assert need <= set(d.columns), f"missing columns: {need - set(d.columns)}"
    for c in ("category", "urgency", "locations"):
        if c not in d:
            d[c] = ""
    d = d.apply(lambda s: s.str.strip())
    problems = []
    unlabeled = d[d.relevant == ""]
    if len(unlabeled):
        problems.append(f"{len(unlabeled)} rows have no `relevant` value (they are skipped)")
    d = d[d.relevant != ""]
    bad_rel = d[~d.relevant.isin(["0", "1"])]
    if len(bad_rel):
        problems.append(f"`relevant` must be 0/1; bad rows: {bad_rel.group_id.tolist()[:10]}")
    d["category"] = d.category.str.upper().replace(CAT_FIX)
    bad_cat = d[(d.relevant == "1") & (d.category != "") & ~d.category.isin(CATEGORY_CODES)]
    if len(bad_cat):
        problems.append(f"unknown categories {sorted(set(bad_cat.category))} in rows {bad_cat.group_id.tolist()[:10]}")
    d.loc[d.relevant == "0", ["category", "urgency"]] = ["", "0"]
    d["locations"] = d.locations.map(norm_locations)
    d.loc[d.relevant == "0", "locations"] = ""  # places only matter for relevant tweets
    no_cat = d[(d.relevant == "1") & (d.category == "")]
    if len(no_cat):
        problems.append(f"{len(no_cat)} relevant rows have no category (kept for relevance, skipped for category): "
                        f"{no_cat.group_id.tolist()}")
    for p in problems:
        print("WARNING:", p)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    d[["group_id", "split", "relevant", "category", "urgency", "locations"]].to_csv(OUT, index=False)
    print(f"saved {len(d)} labels -> {OUT}")
    s = d.assign(relevant=d.relevant.astype(int)).groupby("split").agg(
        n=("relevant", "size"), relevant_share=("relevant", "mean"),
        with_category=("category", lambda x: (x != "").sum()), with_locations=("locations", lambda x: (x != "").sum()))
    print(s.round(2))
    for split, share in s.relevant_share.items():
        if not 0.35 <= share <= 0.65:
            print(f"NOTE: {split} is {share:.0%} relevant; consider topping it up toward ~50/50")


if __name__ == "__main__":
    main(sys.argv[1])
