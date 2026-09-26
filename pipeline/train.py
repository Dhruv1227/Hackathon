"""Train the local model and evaluate it.

  .venv/bin/python -m pipeline.train            # train on everything available, save model
  .venv/bin/python -m pipeline.train --loeo     # also leave-one-event-out relevance check

Uses Gemini labels (data/labels/gemini_labels.csv) for main-dataset TRAIN groups only.
Tune/gold groups are never trained on; gold is the held-out test.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score

from pipeline.context import build_context
from pipeline.model import LocalModel, featurize

ROOT = Path(__file__).resolve().parent.parent
EXT = ROOT / "data/processed/external.csv"
CLEAN = ROOT / "data/processed/clean.csv"
GEMINI = ROOT / "data/labels/gemini_labels.csv"
BONUS_SAMPLE = ROOT / "data/labels/gemini_bonus_sample.csv"   # v4 labels (hazard + country) of a random bonus sample
BONUS_CLEAN = ROOT / "data/processed/bonus_clean.csv"
W_BONUS = 1.0
HAND = ROOT / "data/labels/hand_labels.csv"
# Weights for the main-dataset Gemini labels, chosen on unseen CrisisLex events (leave-one-event-out) + the tune
# set, never on gold. Relevance: 0.3 keeps ranking on new events (the judges' upload) while gaining most of the
# accuracy; higher weights over-fit to this event. Category: 1.0 aligns local categories with Gemini's definitions.
W_EXT, W_MAIN_REL, W_MAIN_CAT = 1.0, 0.3, 1.0


def load_main():
    df = pd.read_csv(CLEAN)
    ctx = build_context(df["text"].tolist())
    canon = df[df.is_canonical].copy()
    canon["masked"] = canon["text"].map(ctx.masker)
    return canon, ctx


def load_bonus_sample():
    """Gemini v4 labels of the random bonus training sample, joined to the masked bonus text."""
    bl = pd.read_csv(BONUS_SAMPLE, keep_default_na=False)
    bc = pd.read_csv(BONUS_CLEAN, keep_default_na=False)
    return bl.merge(bc[["group_id", "masked"]], on="group_id")


def loeo(ext):
    """Leave-one-event-out relevance AUC on CrisisLex: does the model transfer to unseen events?"""
    t26 = ext[ext.source == "crisislext26"].reset_index(drop=True)
    X = featurize(t26["masked"].tolist(), "ext_t26")
    for ev in t26.event.unique():
        tr, te = t26.event != ev, t26.event == ev
        m = LocalModel()
        m.rel.fit(X[tr], t26.relevant[tr])
        p = m.rel.predict_proba(X[te])[:, 1]
        print(f"  LOEO {ev:28s} AUC={roc_auc_score(t26.relevant[te], p):.3f} n={te.sum()}")


def evaluate(model, canon, hand):
    gold = hand[hand.split == "gold"].merge(canon[["group_id", "masked"]], on="group_id")
    if gold.empty:
        return
    pred = model.predict(featurize(gold["masked"].tolist()))
    y = gold["relevant"].astype(int)
    yhat = (pred["p_relevant"] >= 0.5).astype(int)
    print(f"GOLD relevance: acc={accuracy_score(y, yhat):.3f} F1={f1_score(y, yhat):.3f} "
          f"AUC={roc_auc_score(y, pred['p_relevant']):.3f} n={len(gold)}")
    rel = gold[y == 1].index
    has_cat = gold.loc[rel, "category"].fillna("") != ""
    if has_cat.any():
        idx = rel[has_cat.values]
        print(f"GOLD category (relevant only): acc={accuracy_score(gold.loc[idx, 'category'], pred['category'][idx]):.3f} "
              f"n={len(idx)}")


def main(argv):
    ext = pd.read_csv(EXT, keep_default_na=False, na_values={"relevant": [""]})
    if "--loeo" in argv:
        loeo(ext)
    canon, ctx = load_main()
    print(f"region={ctx.region and ctx.region['center']} distinctive_terms={len(ctx.terms)}")

    X_ext = featurize(ext["masked"].tolist(), "ext_all")
    rel_mask = ext["relevant"].notna().values
    cat_mask = (ext["category"] != "").values
    X_rel, y_rel, w_rel = [X_ext[rel_mask]], [ext.relevant[rel_mask].astype(int)], [np.full(rel_mask.sum(), W_EXT)]
    X_cat, y_cat, w_cat = [X_ext[cat_mask]], [ext.category[cat_mask]], [np.full(cat_mask.sum(), W_EXT)]
    haz_mask = (ext["hazard"] != "").values if "hazard" in ext else np.zeros(len(ext), bool)
    X_haz, y_haz, w_haz = [X_ext[haz_mask]], [ext.hazard[haz_mask]], [np.full(haz_mask.sum(), W_EXT)]
    X_urg, y_urg, w_urg = [], [], []

    if GEMINI.exists():
        gl = pd.read_csv(GEMINI, keep_default_na=False).drop(columns=["split"], errors="ignore")
        # the split comes from clean.csv (the source of truth), so tune/gold groups can never slip in
        gl = gl.merge(canon[["group_id", "masked", "split"]], on="group_id")
        gl = gl[gl.split == "train"]
        Xg = featurize(gl["masked"].tolist(), "main_gemini")
        X_rel.append(Xg); y_rel.append(gl.relevant.astype(int)); w_rel.append(np.full(len(gl), W_MAIN_REL))
        r = (gl.relevant.astype(int) == 1).values
        X_cat.append(Xg[r]); y_cat.append(gl.category[r]); w_cat.append(np.full(r.sum(), W_MAIN_CAT))
        X_urg.append(Xg[r]); y_urg.append(gl.urgency[r].astype(int)); w_urg.append(np.ones(r.sum()))
        # the main event is a flood: its relevant tweets are flood examples (same low weight as relevance)
        X_haz.append(Xg[r]); y_haz.append(np.full(r.sum(), "FLOOD")); w_haz.append(np.full(r.sum(), W_MAIN_REL))
        print(f"gemini train labels: {len(gl)} (relevant {r.sum()})")

    if BONUS_SAMPLE.exists() and BONUS_CLEAN.exists():
        bl = load_bonus_sample()
        Xb = featurize(bl["masked"].tolist(), "bonus_sample")
        X_rel.append(Xb); y_rel.append(bl.relevant.astype(int)); w_rel.append(np.full(len(bl), W_BONUS))
        r = (bl.relevant.astype(int) == 1).values
        X_cat.append(Xb[r]); y_cat.append(bl.category[r]); w_cat.append(np.full(r.sum(), W_BONUS))
        X_urg.append(Xb[r]); y_urg.append(bl.urgency[r].astype(int)); w_urg.append(np.ones(r.sum()))
        h = r & (bl.hazard != "").values
        X_haz.append(Xb[h]); y_haz.append(bl.hazard[h]); w_haz.append(np.full(h.sum(), W_BONUS))
        print(f"bonus sample labels: {len(bl)} (relevant {r.sum()}, flood {(bl.hazard[h] == 'FLOOD').sum()})")

    cat = lambda xs: np.concatenate(xs) if xs else None
    model = LocalModel().fit(np.vstack(X_rel), cat(y_rel), cat(w_rel),
                             np.vstack(X_cat), cat(y_cat), cat(w_cat),
                             np.vstack(X_urg) if X_urg else None, cat(y_urg), cat(w_urg),
                             np.vstack(X_haz), cat(y_haz), cat(w_haz))
    model.save()
    print("saved model")
    # gold is deliberately not scored here: use `pipeline.evaluate` (tune) for decisions and
    # `pipeline.evaluate --gold` once for the final report
    if "--gold" in argv and HAND.exists():
        evaluate(model, canon, pd.read_csv(HAND, keep_default_na=False))
    return model


if __name__ == "__main__":
    main(sys.argv[1:])
