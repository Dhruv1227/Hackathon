"""Build the external human-labeled training corpus.

- CrisisLexT26, all 26 events: relevance (any disaster), information type -> category, and the event's hazard
  (FLOOD, QUAKE, ...) for related tweets.
- HumAID (humanitarian categories; all on-topic, so it trains category only).

Leakage guard: every tweet that also appears in a contest dataset (main_contestant.csv, bonus_contestant.csv) is
dropped, and the 2013 Alberta floods event is excluded entirely (it is the main dataset's event). The bonus set
contains ~80% of CrisisLexT26's labeled tweets, so without this the bonus results would be inflated.

Output: data/processed/external.csv with columns
  text, masked, relevant (0/1/NaN), category (code or ""), hazard (code or ""), source, event
"""
import json
from pathlib import Path

import pandas as pd

from pipeline.clean import norm_key
from pipeline.gazetteer import Gazetteer
from pipeline.masking import Masker, distinctive_terms, save_background
from pipeline.taxonomy import EVENT_HAZARD

ROOT = Path(__file__).resolve().parent.parent
CONTEST_FILES = [ROOT / "data/raw/main_contestant.csv", ROOT / "data/raw/bonus_contestant.csv"]
T26 = ROOT / "data/raw/crisislext26/CrisisLexT26"
HUMAID = ROOT / "data/raw/humaid"
OUT = ROOT / "data/processed/external.csv"

T26_EVENTS = [e for e in EVENT_HAZARD if e != "2013_Alberta_floods"]
EXCLUDED = ["2013_Alberta_floods"]
T26_REL = {"Related and informative": 1, "Related - but not informative": 1, "Not related": 0}
T26_CAT = {"Affected individuals": "HUMAN", "Infrastructure and utilities": "INFRA",
           "Donations and volunteering": "AID", "Caution and advice": "ADVISORY",
           "Sympathy and support": "SUPPORT", "Other Useful Information": "OTHER"}
HUMAID_CAT = {"infrastructure_and_utility_damage": "INFRA", "displaced_people_and_evacuations": "EVAC",
              "injured_or_dead_people": "HUMAN", "missing_or_found_people": "HUMAN",
              "requests_or_urgent_needs": "NEEDS", "rescue_volunteering_or_donation_effort": "AID",
              "caution_and_advice": "ADVISORY", "sympathy_and_support": "SUPPORT",
              "other_relevant_information": "OTHER"}
HUMAID_PER_CLASS = 2500  # cap so AID doesn't dominate


def load_t26():
    frames = []
    for ev in T26_EVENTS:
        assert ev not in EXCLUDED
        d = pd.read_csv(T26 / ev / f"{ev}-tweets_labeled.csv", skipinitialspace=True)
        d.columns = [c.strip() for c in d.columns]
        d = d[d["Informativeness"].isin(T26_REL)]
        rel = d["Informativeness"].map(T26_REL)
        frames.append(pd.DataFrame({
            "text": d["Tweet Text"], "relevant": rel,
            "category": d["Information Type"].map(T26_CAT).fillna(""),
            "hazard": [EVENT_HAZARD[ev] if r == 1 else "" for r in rel], "source": "crisislext26", "event": ev}))
    return pd.concat(frames)


def load_humaid():
    rows = [json.loads(l) for f in ("train", "dev", "test") for l in (HUMAID / f"{f}.jsonl").open()]
    d = pd.DataFrame(rows)
    d = d[d["class_label"].isin(HUMAID_CAT)]
    d = d.sample(frac=1, random_state=0).groupby("class_label").head(HUMAID_PER_CLASS)
    return pd.DataFrame({"text": d["tweet_text"], "relevant": float("nan"),
                         "category": d["class_label"].map(HUMAID_CAT), "hazard": "", "source": "humaid",
                         "event": "humaid"})


def main():
    t26, hum = load_t26(), load_humaid()
    df = pd.concat([t26, hum], ignore_index=True)
    df["key"] = df["text"].map(norm_key)
    df = df.drop_duplicates("key")
    contest = set()
    for f in CONTEST_FILES:
        if f.exists():
            contest |= set(pd.read_csv(f)["tweet"].astype(str).map(norm_key))
    leak = df["key"].isin(contest)
    print(f"leakage guard: dropped {int(leak.sum())} tweets that appear in contest data "
          f"({df[leak].groupby('source').size().to_dict()})")
    df = df[~leak].drop(columns="key")
    save_background(df["text"].tolist())

    g = Gazetteer(center=None, world=True)  # region-agnostic: Canadian + world names
    parts = []
    for ev, d in df.groupby("event"):
        terms = distinctive_terms(d["text"].tolist())
        m = Masker(g, terms, proper_nouns=(ev == "humaid"))
        d = d.assign(masked=d["text"].map(m))
        parts.append(d)
        print(f"{ev}: n={len(d)} masked_terms={sorted(terms)[:15]}")
    out = pd.concat(parts)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUT, index=False)
    print(out.groupby("source").agg(n=("text", "size"), rel=("relevant", "mean")))
    print("hazard labels:", out[out.hazard != ""].hazard.value_counts().to_dict())


if __name__ == "__main__":
    main()
