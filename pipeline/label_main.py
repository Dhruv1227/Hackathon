"""Gemini labeling pass over the main dataset, following the evaluation rules.

  tune     label the 100 TUNE tweets (1 call) and score against hand labels. Iterate on the prompt here.
  freeze   record the current prompt hash; after this the gold comparison may run.
  compare  label the 200 GOLD tweets at 50 / 100 / 200 per call (4 + 2 + 1 = 7 calls), score each.
  train    label every TRAIN group at --batch N per call (~72 calls at 100) -> data/labels/gemini_labels.csv
  status   show calls spent so far

Gold tweets are never sent while the prompt is still being tuned (compare refuses without `freeze`).
"""
import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

from pipeline.gemini import Gemini, label_batch, label_prompt, last_known_remaining
from pipeline.metrics import place_match

ROOT = Path(__file__).resolve().parent.parent
CLEAN = ROOT / "data/processed/clean.csv"
HAND = ROOT / "data/labels/hand_labels.csv"
OUT = ROOT / "data/labels/gemini_labels.csv"
FROZEN = ROOT / "data/labels/prompt_frozen.json"
RESULTS = ROOT / "data/labels/batch_size_comparison.json"
EVENT_HINT = "the June 2013 southern Alberta floods (Calgary, High River, Canmore and nearby First Nations)"


def prompt_hash():
    return hashlib.sha256(label_prompt(["<x>"], EVENT_HINT).encode()).hexdigest()[:12]


def canon(split):
    df = pd.read_csv(CLEAN)
    return df[df.is_canonical & (df.split == split)].reset_index(drop=True)


def run(rows, batch, tag, client, retry_missing=True):
    out = {}
    for s in range(0, len(rows), batch):
        chunk = rows.iloc[s:s + batch]
        res = label_batch(client, chunk["text"].tolist(), tag=f"{tag}:b{batch}", event_hint=EVENT_HINT,
                          retry_missing=retry_missing)
        for j, lab in res.items():
            out[int(chunk.iloc[j]["group_id"])] = lab
    return out


def save_side(labels: dict, name: str):
    """Gemini labels for tune/gold groups: shown in the app, never used for training."""
    split = name.split("_")[0]
    pd.DataFrame([{"group_id": g, "split": split, "relevant": lab["relevant"], "confidence": lab["confidence"] / 9,
                   "category": lab["category"], "urgency": lab["urgency"], "locations": ";".join(lab["locations"])}
                  for g, lab in labels.items()]).to_csv(ROOT / f"data/labels/gemini_{name}.csv", index=False)


def score(labels: dict, hand: pd.DataFrame) -> dict:
    h = hand.set_index("group_id")
    ids = [g for g in h.index if g in labels]
    y = h.loc[ids, "relevant"].astype(int)
    yhat = pd.Series([labels[g]["relevant"] for g in ids], index=ids)
    tp = int(((y == 1) & (yhat == 1)).sum()); fp = int(((y == 0) & (yhat == 1)).sum()); fn = int(((y == 1) & (yhat == 0)).sum())
    prec = tp / max(1, tp + fp); rec = tp / max(1, tp + fn)
    both = [g for g in ids if h.at[g, "relevant"] == 1 and labels[g]["relevant"] == 1 and h.at[g, "category"]]
    cat_acc = sum(labels[g]["category"] == h.at[g, "category"] for g in both) / max(1, len(both))
    # place recall/precision on relevant tweets, typo-tolerant, provinces excluded (see pipeline/metrics.py)
    loc_hit = loc_n = loc_pred = 0
    for g in ids:
        if int(h.at[g, "relevant"]) != 1:
            continue
        hit, n_gold, n_pred = place_match(labels[g]["locations"], [s for s in str(h.at[g, "locations"]).split(";") if s])
        loc_hit += hit; loc_n += n_gold; loc_pred += n_pred
    return {"n_labeled": len(ids), "n_missing": len(h) - len(ids), "rel_acc": round(float((y == yhat).mean()), 3),
            "rel_precision": round(prec, 3), "rel_recall": round(rec, 3),
            "rel_f1": round(2 * prec * rec / max(1e-9, prec + rec), 3),
            "cat_acc": round(cat_acc, 3), "cat_n": len(both),
            "loc_recall": round(loc_hit / max(1, loc_n), 3), "loc_precision": round(loc_hit / max(1, loc_pred), 3),
            "loc_n": loc_n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["tune", "freeze", "compare", "train", "status"])
    ap.add_argument("--batch", type=int, default=100)
    ap.add_argument("--max-calls", type=int, default=90)
    a = ap.parse_args()
    hand = pd.read_csv(HAND, keep_default_na=False) if HAND.exists() else pd.DataFrame()

    if a.cmd == "status":
        print(f"proxy quota remaining (as of last call): {last_known_remaining()}")
        print(f"calls spent (ledger): {Gemini.ledger_total()}  failed attempts: {Gemini.ledger_failures()}  prompt={prompt_hash()} "
              f"frozen={json.loads(FROZEN.read_text()) if FROZEN.exists() else None}")
        return

    if a.cmd == "freeze":
        FROZEN.write_text(json.dumps({"prompt_hash": prompt_hash()}))
        print("prompt frozen:", prompt_hash())
        return

    client = Gemini(budget=a.max_calls)
    if client.dry_run:
        raise SystemExit("GEMINI_API_KEY not set (.env). Nothing sent.")

    if a.cmd == "tune":
        rows = canon("tune")
        labels = run(rows, 100, "tune", client, retry_missing=False)
        print(json.dumps(score(labels, hand[hand.split == "tune"]), indent=1))
        save_side(labels, "tune")
        h = hand[hand.split == "tune"].set_index("group_id")
        print("\nDisagreements on relevance (fix the prompt, not the tweets):")
        for _, r in rows.iterrows():
            g = int(r.group_id)
            if g in labels and g in h.index and labels[g]["relevant"] != int(h.at[g, "relevant"]):
                print(f"  hand={h.at[g, 'relevant']} gemini={labels[g]['relevant']} conf={labels[g]['confidence']}  {r.text[:110]}")

    elif a.cmd == "compare":
        if not FROZEN.exists() or json.loads(FROZEN.read_text())["prompt_hash"] != prompt_hash():
            raise SystemExit("Prompt not frozen (or changed since). Run `freeze` after tuning; gold stays unseen until then.")
        rows = canon("gold")
        gold = hand[hand.split == "gold"]
        results = {}
        for b in (50, 100, 200):
            labels = run(rows, b, "gold", client, retry_missing=False)  # measure dropped lines honestly
            results[b] = score(labels, gold)
            print(b, results[b])
            save_side(labels, f"gold_b{b}")
        RESULTS.write_text(json.dumps({"prompt_hash": prompt_hash(), "results": results}, indent=1))
        best = results[100]["rel_acc"]
        ok = [b for b in results if results[b]["rel_acc"] >= best - 0.02 and results[b]["n_missing"] <= 2]
        print(f"\nlargest batch within 2 pts of the 100-batch accuracy and no dropped lines: {max(ok) if ok else 100}")

    elif a.cmd == "train":
        rows = canon("train")
        done = pd.read_csv(OUT, keep_default_na=False) if OUT.exists() else pd.DataFrame(columns=["group_id"])
        rows = rows[~rows.group_id.isin(done.group_id.astype(int))]
        need = -(-len(rows) // a.batch)
        print(f"{len(rows)} train groups left -> {need} calls at batch {a.batch}; ledger so far {Gemini.ledger_total()}")
        labels = run(rows, a.batch, "train", client)
        new = pd.DataFrame([{"group_id": g, "split": "train", "relevant": lab["relevant"], "confidence": lab["confidence"] / 9,
                             "category": lab["category"], "urgency": lab["urgency"], "locations": ";".join(lab["locations"])}
                            for g, lab in labels.items()])
        pd.concat([done, new]).to_csv(OUT, index=False)
        print(f"saved {len(new)} new labels ({len(done) + len(new)} total). calls this run: {client.spent}; "
              f"ledger total: {Gemini.ledger_total()}")


if __name__ == "__main__":
    main()
