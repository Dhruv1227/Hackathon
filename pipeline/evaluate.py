"""Score the local pipeline (no Gemini) against hand labels.

  .venv/bin/python -m pipeline.evaluate            # tune set: use this for decisions
  .venv/bin/python -m pipeline.evaluate --gold     # gold set: report only, don't tune on it
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score

from pipeline.metrics import place_match
from pipeline.model import LocalModel, featurize
from pipeline.train import load_main

ROOT = Path(__file__).resolve().parent.parent
HAND = ROOT / "data/labels/hand_labels.csv"


def main(argv):
    split = "gold" if "--gold" in argv else "tune"
    hand = pd.read_csv(HAND, keep_default_na=False)
    hand = hand[hand.split == split]
    canon, ctx = load_main()
    d = hand.merge(canon[["group_id", "text", "masked"]], on="group_id")
    pred = LocalModel.load().predict(featurize(d["masked"].tolist()))
    y, p = d.relevant.astype(int).values, pred["p_relevant"]
    yhat = (p >= 0.5).astype(int)
    print(f"== {split.upper()} (n={len(d)}, {y.mean():.0%} relevant) — local model only, no Gemini")
    print(f"relevance: acc={accuracy_score(y, yhat):.3f}  P={precision_score(y, yhat):.3f}  R={recall_score(y, yhat):.3f}  "
          f"F1={f1_score(y, yhat):.3f}  AUC={roc_auc_score(y, p):.3f}")
    band = (p >= 0.15) & (p <= 0.85)
    print(f"uncertain band [0.15,0.85]: {band.sum()} tweets, acc inside={accuracy_score(y[band], yhat[band]) if band.any() else float('nan'):.3f}, "
          f"acc outside={accuracy_score(y[~band], yhat[~band]):.3f}")

    rel = (y == 1) & (d.category != "").values
    cat_acc = accuracy_score(d.category[rel], pred["category"][rel])
    print(f"category (relevant, hand-labeled): acc={cat_acc:.3f} n={rel.sum()}")
    print("  confusions:", pd.crosstab(d.category[rel], pred["category"][rel]).stack().loc[lambda s: s > 0]
          .drop(labels=[(c, c) for c in set(d.category[rel])], errors="ignore").sort_values(ascending=False).head(6).to_dict())

    hits = n_gold = n_pred = 0
    hits_all = n_gold_all = 0
    for text, locs, r in zip(d.text, d.locations, y):
        if r != 1:
            continue
        found = [q["name"] for q in ctx.gazetteer.extract(text)]
        gold = [s for s in locs.split(";") if s]
        h, g, n = place_match(found, gold, ctx.gazetteer)
        hits += h; n_gold += g; n_pred += n
        h2, g2, _ = place_match(found, gold, ctx.gazetteer, skip_regions=False)
        hits_all += h2; n_gold_all += g2
    print(f"places (gazetteer only, relevant tweets): recall={hits / max(1, n_gold):.3f} precision={hits / max(1, n_pred):.3f} "
          f"(excl. provinces, gold={n_gold}); incl. provinces recall={hits_all / max(1, n_gold_all):.3f}")

    # cascade simulation (0 calls): Gemini's saved labels for this split replace local ones inside the band
    gem_path = ROOT / ("data/labels/gemini_gold_b200.csv" if split == "gold" else "data/labels/gemini_tune.csv")
    if gem_path.exists():
        gem = pd.read_csv(gem_path, keep_default_na=False).set_index("group_id")
        gem_rel = d.group_id.map(gem["relevant"]).values
        print("cascade (local decides outside band, Gemini inside):")
        for lo, hi in [(0.15, 0.85), (0.05, 0.95), (0.0, 1.01)]:
            band = (p >= lo) & (p <= hi)
            yc = np.where(band, gem_rel, yhat)
            print(f"  band [{lo:.2f},{hi:.2f}]: Gemini reviews {band.mean():.0%} -> acc={accuracy_score(y, yc):.3f} "
                  f"F1={f1_score(y, yc):.3f}")
        gcat = d.group_id.map(gem["category"]).values
        print(f"  Gemini category acc={accuracy_score(d.category[rel], gcat[rel]):.3f}")

    if "-v" in argv:
        d["p"] = p
        wrong = d[(yhat != y)].sort_values("p")
        print("\nrelevance errors:")
        for r in wrong.itertuples():
            print(f"  hand={r.relevant} p={r.p:.2f}  {r.text[:110]}")


if __name__ == "__main__":
    main(sys.argv[1:])
