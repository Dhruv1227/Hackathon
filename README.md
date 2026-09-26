---
title: Living Flood Map
emoji: 🌊
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# Living Flood Map — CE Strategies Hackathon

Turns a CSV of disaster-period tweets into a map of what is happening on the ground: which tweets
are about the flood, what kind of impact they report, how urgent they are, and where.

**Design goal:** Gemini is used only where it adds the most. A free local model labels everything; Gemini reviews
the tweets the local model is least sure about, within a fixed per-upload call budget.

## How it works

```
CSV ─► clean + group copies ─► local model (MiniLM + LR heads) ─► budgeted Gemini review ─► places ─► map
        (retweets join          relevance / category / urgency     most-uncertain first,      gazetteer:
         originals)             on place-masked text               ≤ 20 requests per upload   OSM + CGNDB + aliases,
                                                                                                Nominatim fallback
```

| Step | Module |
|---|---|
| Clean, dedupe, group retweets / quote-RTs / truncated RTs with their original | `pipeline/clean.py` |
| Group-aware split: tune 100 / gold 200 / train; ~50/50 relevance | `pipeline/split.py` |
| Hand labels (CSV or `labeling/app.py`) → validated | `pipeline/import_labels.py` |
| Gemini batch labeling, compact output `id\|rel\|conf\|CAT\|urg\|locs`, cached, call ledger | `pipeline/gemini.py`, `pipeline/label_main.py` |
| External human labels: CrisisLexT26 flood events (Alberta excluded) + HumAID | `pipeline/external.py` |
| Place / event-name masking so the model learns flood language, not Calgary names | `pipeline/masking.py` |
| Local multi-task model: relevance, category, urgency | `pipeline/model.py`, `pipeline/train.py` |
| Gazetteer: CGNDB, OSM neighbourhoods/rivers/bridges/roads for the event region, hashtag aliases (#yyc → Calgary) | `pipeline/gazetteer.py`, `pipeline/osm.py` |
| Event-region detection, incl. non-Canadian events (Boulder → Colorado, not Boulder BC) | `pipeline/context.py` |
| Budgeted cascade for uploads | `pipeline/cascade.py` |
| One-call situation summary of ~150 representative tweets | `pipeline/summary.py` |
| Web app: map, filters, summary, timeline, GeoJSON/CSV export | `app/` |

### Gemini budget

Gemini is reached through the organisers' proxy (`POST /api/generate`, 1,000 requests per key; failed
requests count too). The limit counts requests, not tokens, so tweets are sent 200 per request and answered
in one short line each (`id|1|conf|CAT|urg|place;place`, or `id|0|conf` for noise).

| Use | Requests |
|---|---|
| Prompt tuning on the 100 tune tweets (3 prompt versions + 1 extra from a parser bug, since fixed) | 4 |
| Batch-size comparison on the 200 gold tweets (50/100/200 per request) | 7 |
| Labeling the 7,141 train groups at 200 per request | ~36 |
| Each upload (judges) | ≤ 20 (≈ 4,000 tweets reviewed) |
| Each "Summarize this view" click | 1 |

Safety rails: responses are cached by prompt (re-runs are free); every request is logged with the proxy's
`requests_remaining`; 4xx errors are never retried (429 means the quota is gone) and 5xx only once; uploads fall
back to the local model when ≤ 100 requests remain, summaries when ≤ 10.

Each upload's cap is filled in priority order: (1) the local model's uncertain band, most uncertain first;
(2) confidently-relevant tweets that seem to name a place the gazetteer couldn't match; (3) everything else.
A 2,000-tweet file gets full Gemini coverage; a 20,000-tweet file still can't exceed the cap.

## Results

All numbers are against the hand-labeled sets. Decisions were made on the 100 **tune** tweets; the 200 **gold**
tweets were scored once, after the prompt was frozen.

**Batch size (gold, frozen prompt, dropped lines not re-sent):** 200 per request loses nothing, so it's used.

| Tweets / request | Requests | Dropped lines | Relevance acc | F1 | Category acc | Places recall / precision |
|---|---|---|---|---|---|---|
| 50 | 4 | 1 | 0.960 | 0.961 | 0.55 | 0.85 / 0.81 |
| 100 | 2 | 0 | 0.965 | 0.966 | 0.58 | 0.89 / 0.76 |
| **200** | **1** | **0** | **0.960** | **0.961** | **0.59** | **0.90 / 0.81** |

**Local model alone (no Gemini), trained on external data only — gold:** relevance acc 0.88, F1 0.88, AUC 0.945;
tweets outside the uncertain band are 97% correct, inside it 76% — which is why the band goes to Gemini.
Places from the gazetteer alone: recall 0.86, precision 0.89 (provinces excluded).

Category agreement is ~0.5–0.6 for every method: the nine categories overlap (SUPPORT vs AID vs OTHER), so this
mostly measures agreement with one annotator's reading.

## Evaluation rules followed

- Split **after** dedupe, by group; no group crosses tune / gold / train (asserted in `split.py`).
- Gold (200) is never sent to Gemini while tuning; `label_main compare` refuses until the prompt is frozen.
- Relevance mix kept ~50/50 in each labeled piece.
- CrisisLexT26 2013 Alberta floods (same event as the main dataset) is excluded from training.
- The local model never trains on tune or gold groups.

## Run locally

```bash
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt flask
cp .env.example .env          # add GEMINI_API_KEY (the app runs local-only without it)
.venv/bin/uvicorn app.server:app --port 7860
```

`.env` settings: `GEMINI_API_KEY`, `GEMINI_BASE_URL` (proxy address), `GEMINI_BACKEND=hackathon`,
`GEMINI_MODEL` (empty = proxy default), optional `UPLOAD_CALL_BUDGET` (default 20), `GEMINI_BATCH_SIZE`
(default 200), `GEMINI_QUOTA_RESERVE` (default 100). Check the remaining quota (no request is spent):
`.venv/bin/python -m pipeline.label_main status`.

Rebuild everything from raw data (downloads go to `data/raw/`, gitignored):

```bash
.venv/bin/python -m pipeline.clean && .venv/bin/python -m pipeline.split
.venv/bin/python -m pipeline.gazetteer build          # needs data/raw/cgn/cgn_canada_csv_eng.csv
.venv/bin/python -m pipeline.external                 # needs CrisisLexT26 + HumAID in data/raw/
.venv/bin/python -m pipeline.import_labels data/labels/to_label.csv
.venv/bin/python -m pipeline.label_main tune          # iterate on the prompt
.venv/bin/python -m pipeline.label_main freeze
.venv/bin/python -m pipeline.label_main compare       # 50/100/200 per call on gold
.venv/bin/python -m pipeline.label_main train --batch 200
.venv/bin/python -m pipeline.train --loeo
.venv/bin/python -m pipeline.build_main
```

## Deploy (Hugging Face Spaces, Docker)

Create a Docker Space, add `GEMINI_API_KEY` as a Space **secret** and `GEMINI_BASE_URL` / `GEMINI_BACKEND=hackathon`
as Space variables, then push this repo to it.
The image bakes in the embedding model, the compiled gazetteer, and the precomputed main dataset.

Data: CGNDB © Natural Resources Canada (Open Government Licence – Canada); map data © OpenStreetMap
contributors (ODbL); CrisisLexT26 (Olteanu et al., 2015); HumAID (Alam et al., 2021).
